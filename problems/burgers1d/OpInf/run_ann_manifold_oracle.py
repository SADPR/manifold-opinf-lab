#!/usr/bin/env python3
"""Oracle diagnostics for ANN-manifold continuous-time OpInf ROMs."""

import argparse
import csv
import os
import sys
import tempfile
import time
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ann_manifold_opinf_utils import (
    ann_continuous_feature_matrix,
    ann_manifold_decode,
    load_ann_manifold_model,
    predict_ann_secondary,
    rollout_ann_continuous_rk4,
)
from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import load_or_compute_snaps, param_to_snap_fn, plot_snaps
from manifold_opinf_utils import continuous_feature_matrix, continuous_feature_vector, estimate_time_derivative
from opinf_utils import load_pod_data, project_snapshots
from run_ann_manifold_opinf import DEFAULT_MODEL_PATH
from run_opinf import _relative_error_history, _safe_tag


def _relative_error(hdm_snaps, rom_snaps):
    hdm_snaps = np.asarray(hdm_snaps, dtype=np.float64)
    rom_snaps = np.asarray(rom_snaps, dtype=np.float64)
    finite = np.all(np.isfinite(rom_snaps), axis=0)
    if not np.any(finite):
        return np.inf
    hdm_eval = hdm_snaps[:, finite]
    rom_eval = rom_snaps[:, finite]
    denom = np.linalg.norm(hdm_eval)
    return float(np.linalg.norm(hdm_eval - rom_eval) / (denom if denom > 0.0 else 1.0))


def _relative_matrix_error(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.all(np.isfinite(prediction), axis=0)
    if not np.any(finite):
        return np.inf
    ref_eval = reference[:, finite]
    pred_eval = prediction[:, finite]
    denom = np.linalg.norm(ref_eval)
    return float(np.linalg.norm(ref_eval - pred_eval) / (denom if denom > 0.0 else 1.0))


def _final_relative_error(hdm_snaps, rom_snaps):
    if not np.all(np.isfinite(rom_snaps[:, -1])):
        return np.nan
    denom = np.linalg.norm(hdm_snaps[:, -1])
    return float(np.linalg.norm(hdm_snaps[:, -1] - rom_snaps[:, -1]) / (denom if denom > 0.0 else 1.0))


def _interpolate_columns(values, time_index):
    values = np.asarray(values, dtype=np.float64)
    tau = float(time_index)
    if tau <= 0.0:
        return values[:, 0]
    last = values.shape[1] - 1
    if tau >= last:
        return values[:, last]
    lo = int(np.floor(tau))
    hi = lo + 1
    alpha = tau - lo
    return (1.0 - alpha) * values[:, lo] + alpha * values[:, hi]


def _feature_vector_with_true_qbar(q, qbar, mu, model):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qbar = np.asarray(qbar, dtype=np.float64).reshape(-1)
    theta = continuous_feature_vector(
        q,
        mu,
        feature_mode=model["feature_mode"],
        include_param_linear=bool(model["include_param_linear"]),
        include_quadratic=bool(model["include_quadratic"]),
        include_higher=bool(model["include_higher"]),
        max_degree=int(model["max_degree"]),
    )
    blocks = [theta]
    if bool(model["include_ann_dynamics"]):
        blocks.append(qbar)
        if bool(model["include_param_ann_dynamics"]):
            for val in np.asarray(mu, dtype=np.float64).reshape(-1):
                blocks.append(float(val) * qbar)
    theta = np.concatenate(blocks)
    expected = int(model["operator"].shape[1])
    if theta.size != expected:
        raise ValueError(f"Oracle feature length mismatch: got {theta.size}, expected {expected}.")
    return theta


def _rhs_with_true_qbar(q, qbar, mu, model):
    theta = _feature_vector_with_true_qbar(q, qbar, mu, model)
    theta_scaled = (theta - model["x_mean"]) / model["x_scale"]
    return model["operator"] @ theta_scaled


def _rollout_true_qbar_oracle(q0, qbar_hdm, mu, dt, num_steps, model, max_norm=1e12):
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q_snaps = np.zeros((q0.size, int(num_steps) + 1), dtype=np.float64)
    q_snaps[:, 0] = q0
    q = q0.copy()
    stable_steps = int(num_steps)
    unstable_reason = ""
    h = float(dt)

    for istep in range(int(num_steps)):
        qbar_0 = _interpolate_columns(qbar_hdm, istep)
        qbar_h = _interpolate_columns(qbar_hdm, istep + 0.5)
        qbar_1 = _interpolate_columns(qbar_hdm, istep + 1.0)
        k1 = _rhs_with_true_qbar(q, qbar_0, mu, model)
        k2 = _rhs_with_true_qbar(q + 0.5 * h * k1, qbar_h, mu, model)
        k3 = _rhs_with_true_qbar(q + 0.5 * h * k2, qbar_h, mu, model)
        k4 = _rhs_with_true_qbar(q + h * k3, qbar_1, mu, model)
        q = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (not np.all(np.isfinite(q))) or np.linalg.norm(q) > float(max_norm):
            stable_steps = istep
            unstable_reason = f"unstable at step {istep + 1}"
            q_snaps[:, istep + 1 :] = np.nan
            break
        q_snaps[:, istep + 1] = q
    return q_snaps, stable_steps, unstable_reason


def _rhs_history_with_ann(q_primary, mu, model):
    theta = ann_continuous_feature_matrix(
        q_primary,
        mu,
        model,
        include_ann_dynamics=bool(model["include_ann_dynamics"]),
        include_param_ann_dynamics=bool(model["include_param_ann_dynamics"]),
    )
    theta_scaled = (theta - model["x_mean"][None, :]) / model["x_scale"][None, :]
    return (theta_scaled @ model["operator"].T).T


def _rhs_history_with_true_qbar(q_primary, qbar_hdm, mu, model):
    rhs = np.zeros_like(q_primary, dtype=np.float64)
    for i in range(q_primary.shape[1]):
        rhs[:, i] = _rhs_with_true_qbar(q_primary[:, i], qbar_hdm[:, i], mu, model)
    return rhs


def _plot_multi_error_history(times, histories, out_path, title):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for label, rel_error in histories.items():
        rel_error = np.asarray(rel_error, dtype=np.float64)
        finite = np.isfinite(rel_error)
        if np.any(finite):
            ax.semilogy(times[finite], np.maximum(rel_error[finite], 1e-16), linewidth=1.8, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("relative state error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_metrics_csv(path, metrics):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def main(
    mu1=4.56,
    mu2=0.019,
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    model_path=DEFAULT_MODEL_PATH,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "Oracle"),
    dt=DT,
    num_steps=NUM_STEPS,
    max_norm=1e12,
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    mu = [float(mu1), float(mu2)]
    tag = _safe_tag(mu)

    model = load_ann_manifold_model(model_path, device=device)
    n_primary = int(model["num_primary"])
    n_secondary = int(model["num_secondary"])
    n_total = n_primary + n_secondary

    basis_total, _, u_ref, metadata, basis_path, energy_captured, energy_lost = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=n_total,
    )
    del metadata, energy_captured, energy_lost
    basis_primary = basis_total[:, :n_primary]
    basis_secondary = basis_total[:, n_primary:n_total]

    hdm_path = param_to_snap_fn(mu, snap_folder=snap_folder)
    hdm_was_cached = os.path.exists(hdm_path)
    print("\n====================================================")
    print("        ANN-MANIFOLD OPINF ORACLE DIAGNOSTIC")
    print("====================================================")
    print(f"[ANN-Oracle] mu=({mu[0]:.3f}, {mu[1]:.4f})")
    print(f"[ANN-Oracle] model={model_path}")
    print(f"[ANN-Oracle] HDM snapshot file: {hdm_path}")

    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder)
    elapsed_hdm = time.time() - t0

    q_total = project_snapshots(hdm_snaps, basis_total, u_ref)
    q_primary = q_total[:n_primary, :]
    qbar_hdm = q_total[n_primary:n_total, :]
    q0 = q_primary[:, 0]

    pod_floor_snaps = u_ref[:, None] + basis_total @ q_total
    ann_decoder_true_q_snaps = ann_manifold_decode(
        q_primary,
        basis_primary,
        basis_secondary,
        model,
        u_ref,
        mu,
    )
    qbar_ann_on_true_q = predict_ann_secondary(q_primary, mu, model)

    t0 = time.time()
    q_opinf, stable_steps_opinf, unstable_reason_opinf = rollout_ann_continuous_rk4(
        q0,
        mu,
        dt,
        num_steps,
        model,
        max_norm=max_norm,
    )
    elapsed_opinf = time.time() - t0
    opinf_snaps = ann_manifold_decode(
        q_opinf,
        basis_primary,
        basis_secondary,
        model,
        u_ref,
        mu,
    )

    t0 = time.time()
    q_true_qbar, stable_steps_true_qbar, unstable_reason_true_qbar = _rollout_true_qbar_oracle(
        q0,
        qbar_hdm,
        mu,
        dt,
        num_steps,
        model,
        max_norm=max_norm,
    )
    elapsed_true_qbar = time.time() - t0
    true_qbar_rollout_snaps = (
        u_ref[:, None]
        + basis_primary @ q_true_qbar
        + basis_secondary @ qbar_hdm[:, : int(num_steps) + 1]
    )

    qdot_hdm = estimate_time_derivative(q_primary, dt)
    rhs_ann_on_true_q = _rhs_history_with_ann(q_primary, mu, model)
    rhs_true_qbar_on_true_q = _rhs_history_with_true_qbar(q_primary, qbar_hdm, mu, model)

    metrics = {
        "mu1": float(mu[0]),
        "mu2": float(mu[1]),
        "num_primary": n_primary,
        "num_secondary": n_secondary,
        "pod_floor_state_error": _relative_error(hdm_snaps, pod_floor_snaps),
        "ann_decoder_on_true_q_state_error": _relative_error(hdm_snaps, ann_decoder_true_q_snaps),
        "ann_qbar_on_true_q_relative_error": _relative_matrix_error(qbar_hdm, qbar_ann_on_true_q),
        "opinf_rollout_state_error": _relative_error(hdm_snaps, opinf_snaps),
        "opinf_rollout_final_state_error": _final_relative_error(hdm_snaps, opinf_snaps),
        "true_qbar_rollout_state_error": _relative_error(hdm_snaps, true_qbar_rollout_snaps),
        "true_qbar_rollout_final_state_error": _final_relative_error(hdm_snaps, true_qbar_rollout_snaps),
        "opinf_primary_q_relative_error": _relative_matrix_error(q_primary, q_opinf),
        "true_qbar_primary_q_relative_error": _relative_matrix_error(q_primary, q_true_qbar),
        "rhs_ann_on_true_q_relative_error": _relative_matrix_error(qdot_hdm, rhs_ann_on_true_q),
        "rhs_true_qbar_on_true_q_relative_error": _relative_matrix_error(qdot_hdm, rhs_true_qbar_on_true_q),
        "stable_steps_opinf": int(stable_steps_opinf),
        "stable_steps_true_qbar": int(stable_steps_true_qbar),
        "elapsed_hdm_seconds": float(elapsed_hdm),
        "elapsed_opinf_rollout_seconds": float(elapsed_opinf),
        "elapsed_true_qbar_rollout_seconds": float(elapsed_true_qbar),
    }

    print("[ANN-Oracle] Metrics")
    for key in (
        "pod_floor_state_error",
        "ann_decoder_on_true_q_state_error",
        "ann_qbar_on_true_q_relative_error",
        "opinf_rollout_state_error",
        "true_qbar_rollout_state_error",
        "rhs_ann_on_true_q_relative_error",
        "rhs_true_qbar_on_true_q_relative_error",
    ):
        print(f"[ANN-Oracle]   {key}: {metrics[key]:.6e}")

    np.save(os.path.join(results_dir, f"oracle_q_hdm_{tag}.npy"), q_primary)
    np.save(os.path.join(results_dir, f"oracle_qbar_hdm_{tag}.npy"), qbar_hdm)
    np.save(os.path.join(results_dir, f"oracle_q_opinf_{tag}.npy"), q_opinf)
    np.save(os.path.join(results_dir, f"oracle_q_true_qbar_{tag}.npy"), q_true_qbar)
    np.save(os.path.join(results_dir, f"oracle_snaps_opinf_{tag}.npy"), opinf_snaps)
    np.save(os.path.join(results_dir, f"oracle_snaps_true_qbar_{tag}.npy"), true_qbar_rollout_snaps)

    steps = range(0, int(num_steps) + 1, max(1, int(num_steps) // 5))
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=pod_floor_snaps,
        steps=steps,
        out_path=os.path.join(results_dir, f"oracle_pod_floor_{tag}.png"),
        title=f"HDM/POD floor oracle, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=ann_decoder_true_q_snaps,
        steps=steps,
        out_path=os.path.join(results_dir, f"oracle_ann_decoder_true_q_{tag}.png"),
        title=f"HDM/ANN decoder on true q, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=opinf_snaps,
        steps=steps,
        out_path=os.path.join(results_dir, f"oracle_ann_opinf_rollout_{tag}.png"),
        title=f"HDM/ANN OpInf rollout, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=true_qbar_rollout_snaps,
        steps=steps,
        out_path=os.path.join(results_dir, f"oracle_true_qbar_rollout_{tag}.png"),
        title=f"HDM/true-qbar rollout oracle, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    times = float(dt) * np.arange(int(num_steps) + 1, dtype=np.float64)
    histories = {
        "POD floor": _relative_error_history(hdm_snaps, pod_floor_snaps),
        "ANN decoder on true q": _relative_error_history(hdm_snaps, ann_decoder_true_q_snaps),
        "ANN OpInf rollout": _relative_error_history(hdm_snaps, opinf_snaps),
        "true qbar rollout oracle": _relative_error_history(hdm_snaps, true_qbar_rollout_snaps),
    }
    error_history_path = os.path.join(results_dir, f"oracle_error_histories_{tag}.png")
    _plot_multi_error_history(
        times,
        histories,
        error_history_path,
        title=f"ANN-Manifold oracle error histories, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    metrics_csv_path = os.path.join(results_dir, f"oracle_metrics_{tag}.csv")
    metrics_txt_path = os.path.join(results_dir, f"oracle_summary_{tag}.txt")
    _write_metrics_csv(metrics_csv_path, metrics)
    with open(metrics_txt_path, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: OpInf/run_ann_manifold_oracle.py\n\n")
        file.write("[configuration]\n")
        file.write(f"model_npz: {model_path}\n")
        file.write(f"pod_basis_path: {basis_path}\n")
        file.write(f"hdm_snapshot_file: {hdm_path}\n")
        file.write(f"hdm_loaded_from_cache: {bool(hdm_was_cached)}\n")
        file.write(f"dt: {float(dt):.8e}\n")
        file.write(f"num_steps: {int(num_steps)}\n")
        file.write(f"num_cells: {int(NUM_CELLS)}\n")
        file.write(f"time_scheme: {TIME_SCHEME}\n")
        file.write(f"unstable_reason_opinf: {unstable_reason_opinf or 'none'}\n")
        file.write(f"unstable_reason_true_qbar: {unstable_reason_true_qbar or 'none'}\n\n")
        file.write("[metrics]\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                file.write(f"{key}: {value:.8e}\n")
            else:
                file.write(f"{key}: {value}\n")
        file.write("\n[outputs]\n")
        file.write(f"metrics_csv: {metrics_csv_path}\n")
        file.write(f"error_histories_png: {error_history_path}\n")
        file.write(f"summary_txt: {metrics_txt_path}\n")

    print(f"[ANN-Oracle] Metrics CSV saved: {metrics_csv_path}")
    print(f"[ANN-Oracle] Error histories saved: {error_history_path}")
    print(f"[ANN-Oracle] Summary saved: {metrics_txt_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ANN-manifold OpInf oracle diagnostics for one test point.")
    parser.add_argument("--mu1", type=float, default=4.56)
    parser.add_argument("--mu2", type=float, default=0.019)
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "Oracle"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-norm", type=float, default=1e12)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(
        mu1=args.mu1,
        mu2=args.mu2,
        pod_dir=args.pod_dir,
        model_path=args.model_path,
        snap_folder=args.snap_folder,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
        max_norm=args.max_norm,
        device=args.device,
    )
