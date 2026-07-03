#!/usr/bin/env python3
"""Run one linear discrete-time OpInf ROM case."""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import load_or_compute_snaps, param_to_snap_fn, plot_snaps
import matplotlib.pyplot as plt
from opinf_utils import (
    load_model,
    load_pod_data,
    project_snapshots,
    reconstruct_snapshots,
    rollout_linear_discrete,
)


DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "linear_discrete_n20.npz")


def _safe_tag(mu):
    return f"mu1_{float(mu[0]):.2f}_mu2_{float(mu[1]):.3f}"


def _relative_error_history(hdm_snaps, rom_snaps):
    hdm_snaps = np.asarray(hdm_snaps, dtype=np.float64)
    rom_snaps = np.asarray(rom_snaps, dtype=np.float64)
    if hdm_snaps.shape != rom_snaps.shape:
        raise ValueError(f"Snapshot shape mismatch: HDM {hdm_snaps.shape}, ROM {rom_snaps.shape}.")

    rel = np.full(hdm_snaps.shape[1], np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(rom_snaps), axis=0)
    if np.any(finite):
        err = np.linalg.norm(hdm_snaps[:, finite] - rom_snaps[:, finite], axis=0)
        denom = np.linalg.norm(hdm_snaps[:, finite], axis=0)
        rel[finite] = err / np.where(denom > 0.0, denom, 1.0)
    return rel


def _plot_error_history(times, rel_error, out_path, title):
    times = np.asarray(times, dtype=np.float64)
    rel_error = np.asarray(rel_error, dtype=np.float64)
    finite = np.isfinite(rel_error)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if np.any(finite):
        ax.semilogy(times[finite], np.maximum(rel_error[finite], 1e-16), color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("time")
    ax.set_ylabel("relative state error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(
    mu1=4.56,
    mu2=0.019,
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    model_path=DEFAULT_MODEL_PATH,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf"),
    dt=DT,
    num_steps=NUM_STEPS,
    max_norm=1e12,
):
    os.makedirs(results_dir, exist_ok=True)
    mu = [float(mu1), float(mu2)]

    model = load_model(model_path)
    n_keep = int(model["num_modes"])
    basis, _, u_ref, metadata, basis_path, energy_captured, energy_lost = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=n_keep,
    )
    del metadata

    hdm_path = param_to_snap_fn(mu, snap_folder=snap_folder)
    hdm_was_cached = os.path.exists(hdm_path)
    print(f"[OpInf] HDM snapshot file: {hdm_path}")
    if hdm_was_cached:
        print("[OpInf] HDM snapshots found. Loading before OpInf rollout.")
    else:
        print("[OpInf] HDM snapshots not found. Computing before OpInf rollout.")

    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder)
    elapsed_hdm = time.time() - t0

    q_hdm = project_snapshots(hdm_snaps, basis, u_ref)
    q0 = q_hdm[:, 0]

    print(f"[OpInf] Loaded model: {model_path}")
    print(f"[OpInf] POD basis: {basis_path}")
    print(f"[OpInf] Basis size used: {n_keep}")
    print(f"[OpInf] Training one-step error: {model['train_relative_one_step_error']:.3e}")

    t0 = time.time()
    q_rom, stable_steps, unstable_reason = rollout_linear_discrete(
        q0,
        mu,
        num_steps,
        operator=model["operator"],
        x_mean=model["x_mean"],
        x_scale=model["x_scale"],
        feature_mode=model["feature_mode"],
        max_norm=max_norm,
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
    rom_path = os.path.join(results_dir, f"opinf_linear_discrete_{tag}.npy")
    q_path = os.path.join(results_dir, f"opinf_linear_discrete_q_{tag}.npy")
    err_path = os.path.join(results_dir, f"opinf_linear_discrete_error_history_{tag}.npy")
    np.save(rom_path, rom_snaps)
    np.save(q_path, q_rom)

    fig_path = os.path.join(results_dir, f"opinf_linear_discrete_{tag}.png")
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=rom_snaps,
        steps=range(0, int(num_steps) + 1, max(1, int(num_steps) // 5)),
        out_path=fig_path,
        title=f"HDM/Linear OpInf 1D Burgers, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    rel_error_history = _relative_error_history(hdm_snaps, rom_snaps)
    np.save(err_path, rel_error_history)
    error_fig_path = os.path.join(results_dir, f"opinf_linear_discrete_error_{tag}.png")
    times = float(dt) * np.arange(int(num_steps) + 1, dtype=np.float64)
    _plot_error_history(
        times,
        rel_error_history,
        error_fig_path,
        title=f"Linear OpInf error history, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    report_path = os.path.join(results_dir, f"opinf_linear_discrete_summary_{tag}.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: OpInf/run_opinf.py\n\n")
        file.write("[configuration]\n")
        file.write(f"model_family: linear_discrete_parametric\n")
        file.write(f"mu1: {mu[0]:.8e}\n")
        file.write(f"mu2: {mu[1]:.8e}\n")
        file.write(f"dt: {float(dt):.8e}\n")
        file.write(f"num_steps: {int(num_steps)}\n")
        file.write(f"num_cells: {int(NUM_CELLS)}\n")
        file.write(f"time_scheme: {TIME_SCHEME}\n")
        file.write(f"num_modes: {int(n_keep)}\n")
        file.write(f"ridge: {model['ridge']:.8e}\n")
        file.write(f"feature_mode: {model['feature_mode']}\n")
        file.write(f"model_npz: {model_path}\n")
        file.write(f"pod_basis_path: {basis_path}\n")
        file.write(f"hdm_snapshot_file: {hdm_path}\n")
        file.write(f"hdm_loaded_from_cache: {bool(hdm_was_cached)}\n")
        file.write("\n[results]\n")
        file.write(f"relative_error_all_finite_snapshots: {rel_err:.8e}\n")
        file.write(f"relative_error_final_snapshot: {final_rel_err:.8e}\n")
        file.write(f"stable_steps: {int(stable_steps)}\n")
        file.write(f"unstable_reason: {unstable_reason or 'none'}\n")
        file.write(f"train_relative_one_step_error: {model['train_relative_one_step_error']:.8e}\n")
        file.write(f"elapsed_opinf_seconds: {elapsed_rom:.8e}\n")
        file.write(f"elapsed_hdm_before_opinf_seconds: {elapsed_hdm:.8e}\n")
        if energy_captured is not None:
            file.write(f"energy_captured_used_modes: {energy_captured:.8e}\n")
            file.write(f"energy_lost_used_modes: {energy_lost:.8e}\n")
        file.write("\n[outputs]\n")
        file.write(f"rom_snapshots_npy: {rom_path}\n")
        file.write(f"rom_coordinates_npy: {q_path}\n")
        file.write(f"error_history_npy: {err_path}\n")
        file.write(f"snapshot_overlay_png: {fig_path}\n")
        file.write(f"error_history_png: {error_fig_path}\n")
        file.write(f"summary_txt: {report_path}\n")

    print(f"[OpInf] Relative error over finite snapshots: {rel_err:.3e}")
    if np.isfinite(final_rel_err):
        print(f"[OpInf] Final snapshot relative error: {final_rel_err:.3e}")
    else:
        print(f"[OpInf] Final snapshot relative error: NaN ({unstable_reason})")
    print(f"[OpInf] Summary saved: {report_path}")
    print(f"[OpInf] Snapshot overlay saved: {fig_path}")
    print(f"[OpInf] Error plot saved: {error_fig_path}")
    return rom_snaps, rel_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one linear discrete-time OpInf ROM test point.")
    parser.add_argument("--mu1", type=float, default=4.56)
    parser.add_argument("--mu2", type=float, default=0.019)
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf"))
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
