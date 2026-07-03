#!/usr/bin/env python3
"""Run one standard linear-subspace continuous-time OpInf ROM case."""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LAB_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
for path in (PROJECT_ROOT, LAB_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import load_or_compute_snaps, param_to_snap_fn, plot_snaps
from manifold_opinf_utils import continuous_feature_vector
from opinf_lab.time_integration import rollout_rk4
from opinf_utils import load_pod_data, project_snapshots, reconstruct_snapshots
from run_opinf import _plot_error_history, _relative_error_history, _safe_tag


DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "standard_continuous_quadratic_r20.npz")


def load_standard_continuous_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Standard continuous OpInf model not found: {model_path}")
    data = np.load(model_path, allow_pickle=False)
    model = {key: data[key] for key in data.files}
    for key in ("model_family", "feature_mode", "pod_basis_path"):
        if key in model:
            model[key] = str(np.asarray(model[key]).item())
    for key in ("num_modes", "num_steps", "max_degree", "num_features", "rk4_substeps"):
        if key in model:
            model[key] = int(np.asarray(model[key]).item())
    for key in (
        "dt",
        "dynamics_ridge",
        "relative_derivative_training_error",
        "validation_rollout_error",
        "training_rollout_error",
        "energy_captured",
    ):
        if key in model:
            model[key] = float(np.asarray(model[key]).item())
    for key in ("include_param_linear", "include_quadratic", "include_higher"):
        if key in model:
            model[key] = bool(int(np.asarray(model[key]).item()))
    model["operator"] = np.asarray(model["operator"], dtype=np.float64)
    model["x_mean"] = np.asarray(model["x_mean"], dtype=np.float64)
    model["x_scale"] = np.asarray(model["x_scale"], dtype=np.float64)
    return model


def rhs(q, mu, model):
    theta = continuous_feature_vector(
        q,
        mu,
        feature_mode=model["feature_mode"],
        include_param_linear=bool(model["include_param_linear"]),
        include_quadratic=bool(model["include_quadratic"]),
        include_higher=bool(model["include_higher"]),
        max_degree=int(model["max_degree"]),
    )
    return model["operator"] @ ((theta - model["x_mean"]) / model["x_scale"])


def main(
    mu1=4.56,
    mu2=0.019,
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    model_path=DEFAULT_MODEL_PATH,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-Standard", "Continuous"),
    dt=DT,
    num_steps=NUM_STEPS,
    max_norm=1e12,
):
    os.makedirs(results_dir, exist_ok=True)
    mu = [float(mu1), float(mu2)]
    model = load_standard_continuous_model(model_path)
    n_modes = int(model["num_modes"])
    basis, _, u_ref, metadata, basis_path, energy_captured, energy_lost = load_pod_data(
        pod_dir, U0.size, num_modes=n_modes
    )
    del metadata, energy_captured, energy_lost

    hdm_path = param_to_snap_fn(mu, snap_folder=snap_folder)
    hdm_was_cached = os.path.exists(hdm_path)
    print(f"[Standard-Cont] HDM snapshot file: {hdm_path}")
    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder)
    elapsed_hdm = time.time() - t0

    q_hdm = project_snapshots(hdm_snaps, basis, u_ref)
    q0 = q_hdm[:, 0]

    print(f"[Standard-Cont] Loaded model: {model_path}")
    print(f"[Standard-Cont] POD basis: {basis_path}")
    print(f"[Standard-Cont] modes r={n_modes}")
    print(f"[Standard-Cont] derivative training error: {model['relative_derivative_training_error']:.3e}")
    print(f"[Standard-Cont] RK4 substeps: {int(model.get('rk4_substeps', 1))}")

    t0 = time.time()
    q_rom, stable_steps, unstable_reason = rollout_rk4(
        lambda q: rhs(q, mu, model),
        q0,
        dt,
        num_steps,
        max_norm=max_norm,
        substeps=int(model.get("rk4_substeps", 1)),
    )
    elapsed_rom = time.time() - t0
    rom_snaps = reconstruct_snapshots(q_rom, basis, u_ref)

    finite_mask = np.all(np.isfinite(rom_snaps), axis=0)
    if not np.all(finite_mask):
        last = int(np.flatnonzero(finite_mask)[-1]) if np.any(finite_mask) else 0
        hdm_eval = hdm_snaps[:, : last + 1]
        rom_eval = rom_snaps[:, : last + 1]
    else:
        hdm_eval = hdm_snaps
        rom_eval = rom_snaps
    denom = np.linalg.norm(hdm_eval)
    rel_err = np.linalg.norm(hdm_eval - rom_eval) / (denom if denom > 0.0 else 1.0)
    final_rel_err = np.nan
    if np.all(np.isfinite(rom_snaps[:, -1])):
        final_rel_err = np.linalg.norm(hdm_snaps[:, -1] - rom_snaps[:, -1]) / (
            np.linalg.norm(hdm_snaps[:, -1]) or 1.0
        )

    tag = _safe_tag(mu)
    rom_path = os.path.join(results_dir, f"standard_continuous_snaps_{tag}.npy")
    q_path = os.path.join(results_dir, f"standard_continuous_q_{tag}.npy")
    err_path = os.path.join(results_dir, f"standard_continuous_error_history_{tag}.npy")
    np.save(rom_path, rom_snaps)
    np.save(q_path, q_rom)
    rel_error_history = _relative_error_history(hdm_snaps, rom_snaps)
    np.save(err_path, rel_error_history)

    fig_path = os.path.join(results_dir, f"standard_continuous_{tag}.png")
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=rom_snaps,
        steps=range(0, int(num_steps) + 1, max(1, int(num_steps) // 5)),
        out_path=fig_path,
        title=f"HDM/Standard continuous OpInf 1D Burgers, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )
    error_fig_path = os.path.join(results_dir, f"standard_continuous_error_{tag}.png")
    times = float(dt) * np.arange(int(num_steps) + 1, dtype=np.float64)
    _plot_error_history(
        times,
        rel_error_history,
        error_fig_path,
        title=f"Standard continuous OpInf error history, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    report_path = os.path.join(results_dir, f"standard_continuous_summary_{tag}.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: OpInf/run_standard_continuous_opinf.py\n\n")
        file.write("[configuration]\n")
        file.write(f"model_family: {model['model_family']}\n")
        file.write(f"mu1: {mu[0]:.8e}\n")
        file.write(f"mu2: {mu[1]:.8e}\n")
        file.write(f"dt: {float(dt):.8e}\n")
        file.write(f"num_steps: {int(num_steps)}\n")
        file.write(f"num_cells: {int(NUM_CELLS)}\n")
        file.write(f"time_scheme: {TIME_SCHEME}\n")
        file.write(f"num_modes: {n_modes}\n")
        file.write(f"dynamics_ridge: {model['dynamics_ridge']:.8e}\n")
        file.write(f"include_param_linear: {bool(model['include_param_linear'])}\n")
        file.write(f"include_quadratic: {bool(model['include_quadratic'])}\n")
        file.write(f"include_higher: {bool(model['include_higher'])}\n")
        file.write(f"num_features: {model['num_features']}\n")
        file.write(f"rk4_substeps: {int(model.get('rk4_substeps', 1))}\n")
        file.write(f"model_npz: {model_path}\n")
        file.write(f"pod_basis_path: {basis_path}\n")
        file.write(f"hdm_snapshot_file: {hdm_path}\n")
        file.write(f"hdm_loaded_from_cache: {bool(hdm_was_cached)}\n")
        file.write("\n[results]\n")
        file.write(f"relative_error_all_finite_snapshots: {rel_err:.8e}\n")
        file.write(f"relative_error_final_snapshot: {final_rel_err:.8e}\n")
        file.write(f"stable_steps: {int(stable_steps)}\n")
        file.write(f"unstable_reason: {unstable_reason or 'none'}\n")
        file.write(f"elapsed_opinf_seconds: {elapsed_rom:.8e}\n")
        file.write(f"elapsed_hdm_before_opinf_seconds: {elapsed_hdm:.8e}\n")
        file.write("\n[outputs]\n")
        file.write(f"rom_snapshots_npy: {rom_path}\n")
        file.write(f"rom_coordinates_npy: {q_path}\n")
        file.write(f"error_history_npy: {err_path}\n")
        file.write(f"snapshot_overlay_png: {fig_path}\n")
        file.write(f"error_history_png: {error_fig_path}\n")
        file.write(f"summary_txt: {report_path}\n")

    print(f"[Standard-Cont] Relative error over finite snapshots: {rel_err:.3e}")
    print(f"[Standard-Cont] Summary saved: {report_path}")
    return rom_snaps, rel_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one standard continuous OpInf test point.")
    parser.add_argument("--mu1", type=float, default=4.56)
    parser.add_argument("--mu2", type=float, default=0.019)
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-Standard", "Continuous"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-norm", type=float, default=1e12)
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
    )
