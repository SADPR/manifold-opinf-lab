#!/usr/bin/env python3
"""Run one POD-based quadratic-manifold continuous-time OpInf ROM case."""

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
from manifold_opinf_utils import load_manifold_model, manifold_decode, rollout_continuous_rk4
from opinf_utils import load_pod_data, project_snapshots
from run_opinf import _plot_error_history, _relative_error_history, _safe_tag


DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "manifold_continuous_r20_q40.npz")


def main(
    mu1=4.56,
    mu2=0.019,
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    model_path=DEFAULT_MODEL_PATH,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-Manifold"),
    dt=DT,
    num_steps=NUM_STEPS,
    max_norm=1e12,
):
    os.makedirs(results_dir, exist_ok=True)
    mu = [float(mu1), float(mu2)]
    model = load_manifold_model(model_path)
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
    print(f"[MPOD-OpInf] HDM snapshot file: {hdm_path}")
    if hdm_was_cached:
        print("[MPOD-OpInf] HDM snapshots found. Loading before manifold OpInf rollout.")
    else:
        print("[MPOD-OpInf] HDM snapshots not found. Computing before manifold OpInf rollout.")

    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder)
    elapsed_hdm = time.time() - t0

    q_hdm = project_snapshots(hdm_snaps, basis_primary, u_ref)
    q0 = q_hdm[:, 0]

    print(f"[MPOD-OpInf] Loaded model: {model_path}")
    print(f"[MPOD-OpInf] POD basis: {basis_path}")
    print(f"[MPOD-OpInf] Primary modes r={n_primary}, secondary modes q={n_secondary}")
    print(f"[MPOD-OpInf] Manifold training error: {model['relative_manifold_training_error']:.3e}")
    print(f"[MPOD-OpInf] Derivative training error: {model['relative_derivative_training_error']:.3e}")

    t0 = time.time()
    q_rom, stable_steps, unstable_reason = rollout_continuous_rk4(
        q0,
        mu,
        dt,
        num_steps,
        model,
        max_norm=max_norm,
    )
    elapsed_rom = time.time() - t0
    rom_snaps = manifold_decode(
        q_rom,
        basis_primary,
        basis_secondary,
        model["xi"],
        u_ref,
        polynomial_order=model["polynomial_order"],
    )

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
    rom_path = os.path.join(results_dir, f"mpod_opinf_snaps_{tag}.npy")
    q_path = os.path.join(results_dir, f"mpod_opinf_q_{tag}.npy")
    err_path = os.path.join(results_dir, f"mpod_opinf_error_history_{tag}.npy")
    np.save(rom_path, rom_snaps)
    np.save(q_path, q_rom)

    fig_path = os.path.join(results_dir, f"mpod_opinf_{tag}.png")
    plot_snaps(
        GRID_X,
        hdm_snaps,
        rom_snaps=rom_snaps,
        steps=range(0, int(num_steps) + 1, max(1, int(num_steps) // 5)),
        out_path=fig_path,
        title=f"HDM/MPOD-OpInf 1D Burgers, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    rel_error_history = _relative_error_history(hdm_snaps, rom_snaps)
    np.save(err_path, rel_error_history)
    error_fig_path = os.path.join(results_dir, f"mpod_opinf_error_{tag}.png")
    times = float(dt) * np.arange(int(num_steps) + 1, dtype=np.float64)
    _plot_error_history(
        times,
        rel_error_history,
        error_fig_path,
        title=f"MPOD-OpInf error history, mu=({mu[0]:.2f}, {mu[1]:.3f})",
    )

    report_path = os.path.join(results_dir, f"mpod_opinf_summary_{tag}.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: OpInf/run_manifold_opinf.py\n\n")
        file.write("[configuration]\n")
        file.write("model_family: mpod_quadratic_manifold_continuous\n")
        file.write(f"mu1: {mu[0]:.8e}\n")
        file.write(f"mu2: {mu[1]:.8e}\n")
        file.write(f"dt: {float(dt):.8e}\n")
        file.write(f"num_steps: {int(num_steps)}\n")
        file.write(f"num_cells: {int(NUM_CELLS)}\n")
        file.write(f"time_scheme: {TIME_SCHEME}\n")
        file.write(f"num_primary: {n_primary}\n")
        file.write(f"num_secondary: {n_secondary}\n")
        file.write(f"polynomial_order: {model['polynomial_order']}\n")
        file.write(f"num_features: {model['num_features']}\n")
        file.write(f"manifold_ridge: {model['manifold_ridge']:.8e}\n")
        file.write(f"dynamics_ridge: {model['dynamics_ridge']:.8e}\n")
        file.write(f"include_param_linear: {bool(model['include_param_linear'])}\n")
        file.write(f"include_quadratic: {bool(model['include_quadratic'])}\n")
        file.write(f"include_higher: {bool(model['include_higher'])}\n")
        file.write(f"include_manifold_dynamics: {bool(model.get('include_manifold_dynamics', False))}\n")
        file.write(f"max_degree: {model['max_degree']}\n")
        file.write(f"model_npz: {model_path}\n")
        file.write(f"pod_basis_path: {basis_path}\n")
        file.write(f"hdm_snapshot_file: {hdm_path}\n")
        file.write(f"hdm_loaded_from_cache: {bool(hdm_was_cached)}\n")
        file.write("\n[results]\n")
        file.write(f"relative_error_all_finite_snapshots: {rel_err:.8e}\n")
        file.write(f"relative_error_final_snapshot: {final_rel_err:.8e}\n")
        file.write(f"stable_steps: {int(stable_steps)}\n")
        file.write(f"unstable_reason: {unstable_reason or 'none'}\n")
        file.write(f"relative_manifold_training_error: {model['relative_manifold_training_error']:.8e}\n")
        file.write(f"relative_derivative_training_error: {model['relative_derivative_training_error']:.8e}\n")
        file.write(f"elapsed_opinf_seconds: {elapsed_rom:.8e}\n")
        file.write(f"elapsed_hdm_before_opinf_seconds: {elapsed_hdm:.8e}\n")
        file.write("\n[outputs]\n")
        file.write(f"rom_snapshots_npy: {rom_path}\n")
        file.write(f"rom_coordinates_npy: {q_path}\n")
        file.write(f"error_history_npy: {err_path}\n")
        file.write(f"snapshot_overlay_png: {fig_path}\n")
        file.write(f"error_history_png: {error_fig_path}\n")
        file.write(f"summary_txt: {report_path}\n")

    print(f"[MPOD-OpInf] Relative error over finite snapshots: {rel_err:.3e}")
    if np.isfinite(final_rel_err):
        print(f"[MPOD-OpInf] Final snapshot relative error: {final_rel_err:.3e}")
    else:
        print(f"[MPOD-OpInf] Final snapshot relative error: NaN ({unstable_reason})")
    print(f"[MPOD-OpInf] Summary saved: {report_path}")
    print(f"[MPOD-OpInf] Snapshot overlay saved: {fig_path}")
    print(f"[MPOD-OpInf] Error plot saved: {error_fig_path}")
    return rom_snaps, rel_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one MPOD-OpInf test point.")
    parser.add_argument("--mu1", type=float, default=4.56)
    parser.add_argument("--mu2", type=float, default=0.019)
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-Manifold"))
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
