#!/usr/bin/env python3
"""Run alternating-minimization manifold OpInf for the KdV soliton."""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from kdv.core import write_txt_report
from mam_opinf_utils import (
    load_model,
    reconstruct_mam_snapshots,
    solve_mam_coordinates,
)
from mpod_opinf_utils import (
    load_fom_dataset,
    plot_error_history,
    plot_snapshot_comparison,
    plot_spacetime_comparison,
    relative_error_history,
    relative_state_error,
    rollout_rk4,
)


def main(
    model_path=os.path.join(SCRIPT_DIR, "models", "mam_opinf_r5_p2_q9.npz"),
    snapshot_file=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "MAM"),
    max_norm=1e6,
):
    model = load_model(model_path)
    if snapshot_file is None:
        snapshot_file = model.get("snapshot_file", os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    os.makedirs(results_dir, exist_ok=True)

    x, times, snapshots, train_mask, _ = load_fom_dataset(snapshot_file)
    basis = np.asarray(model["basis"], dtype=np.float64)
    basis_bar = np.asarray(model["basis_bar"], dtype=np.float64)
    u_ref = np.asarray(model["u_ref"], dtype=np.float64)
    xi = np.asarray(model["xi"], dtype=np.float64)
    coeffs = np.asarray(model["coeffs"], dtype=np.float64)
    exponents = np.asarray(model["exponents"], dtype=np.int64)
    num_modes = int(model["num_modes"])
    num_secondary = int(model["num_secondary"])
    degree = int(model["degree"])
    train_final_time = float(model["train_final_time"])
    rk4_substeps = int(model.get("rk4_substeps", 1))
    q_initial_guess = np.asarray(model.get("q_initial", basis.T @ (snapshots[:, 0] - u_ref)), dtype=np.float64)

    q0, q0_result = solve_mam_coordinates(
        snapshots[:, 0],
        q_initial_guess,
        basis,
        basis_bar,
        xi,
        u_ref,
        degree,
        ls_ftol=float(model.get("ls_ftol", 1e-9)),
    )
    q_rom, unstable, unstable_index = rollout_rk4(
        q0,
        times,
        coeffs,
        exponents,
        max_norm=max_norm,
        substeps=rk4_substeps,
    )
    rom = reconstruct_mam_snapshots(q_rom, basis, basis_bar, xi, u_ref, degree)

    prediction_mask = times >= train_final_time - 1e-14
    train_error = relative_state_error(snapshots[:, train_mask], rom[:, train_mask], u_ref=u_ref)
    prediction_error = relative_state_error(snapshots[:, prediction_mask], rom[:, prediction_mask], u_ref=u_ref)
    full_error = relative_state_error(snapshots, rom, u_ref=u_ref)
    error_history = relative_error_history(snapshots, rom)

    tag = f"r{num_modes}_p{degree}_q{num_secondary}"
    spacetime_plot = os.path.join(results_dir, f"mam_opinf_{tag}_spacetime.png")
    snapshot_plot = os.path.join(results_dir, f"mam_opinf_{tag}_snapshots.png")
    error_plot = os.path.join(results_dir, f"mam_opinf_{tag}_error_history.png")
    q_path = os.path.join(results_dir, f"mam_opinf_{tag}_q_rollout.npz")
    summary_path = os.path.join(results_dir, f"mam_opinf_{tag}_summary.txt")

    plot_spacetime_comparison(
        x,
        times,
        snapshots,
        rom,
        train_final_time,
        spacetime_plot,
        title=f"KdV MAM-OpInf, r={num_modes}, p={degree}, q={num_secondary}",
        rom_label="MAM-OpInf",
    )
    plot_snapshot_comparison(
        x,
        times,
        snapshots,
        rom,
        snapshot_plot,
        snapshot_times=(train_final_time, times[-1]),
        rom_label="MAM-OpInf",
    )
    plot_error_history(
        times,
        error_history,
        train_final_time,
        error_plot,
        title=f"MAM-OpInf error history, r={num_modes}, p={degree}, q={num_secondary}",
    )
    np.savez(
        q_path,
        times=times,
        q0=q0,
        q_initial_guess=q_initial_guess,
        q0_cost=np.asarray(float(q0_result.cost), dtype=np.float64),
        q0_nfev=np.asarray(int(q0_result.nfev), dtype=np.int64),
        q_rom=q_rom,
        error_history=error_history,
        unstable=np.asarray(int(bool(unstable)), dtype=np.int64),
        unstable_index=np.asarray(int(unstable_index), dtype=np.int64),
    )

    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/run_mam_opinf.py")]),
            (
                "model",
                [
                    ("model_path", model_path),
                    ("snapshot_file", snapshot_file),
                    ("num_modes_r", num_modes),
                    ("num_secondary_q", num_secondary),
                    ("total_modes_r_plus_q", int(model["total_modes"])),
                    ("polynomial_degree_p", degree),
                    ("gamma", model["gamma"]),
                    ("am_tol", model["am_tol"]),
                    ("am_iterations", model["am_iterations"]),
                    ("am_converged", model["am_converged"]),
                    ("ridge_c", model["ridge_c"]),
                    ("ridge_a", model["ridge_a"]),
                    ("ridge_h", model["ridge_h"]),
                    ("ridge_p", model["ridge_p"]),
                    ("rk4_substeps", rk4_substeps),
                    ("regularizer_convention", model.get("regularizer_convention", "opinf_direct_weight")),
                    ("model_variant", model.get("model_variant", "standard_mam_opinf")),
                    ("energy_captured_metric", model["energy_captured"]),
                    ("relative_reconstruction_error", model["relative_reconstruction_error"]),
                    ("relative_derivative_error", model["relative_derivative_error"]),
                    ("training_rollout_error_selected", model["training_rollout_error"]),
                    ("q0_nonlinear_ls_cost", float(q0_result.cost)),
                    ("q0_nonlinear_ls_nfev", int(q0_result.nfev)),
                ],
            ),
            (
                "rollout",
                [
                    ("full_window_error", full_error),
                    ("training_window_error", train_error),
                    ("prediction_window_error", prediction_error),
                    ("unstable", bool(unstable)),
                    ("unstable_index", int(unstable_index)),
                    ("num_time_samples", int(times.size)),
                ],
            ),
            (
                "outputs",
                [
                    ("spacetime_png", spacetime_plot),
                    ("snapshots_png", snapshot_plot),
                    ("error_history_png", error_plot),
                    ("q_rollout_npz", q_path),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )

    print("\n====================================================")
    print("              KDV MAM-OPINF ROLLOUT")
    print("====================================================")
    print(f"[KDV-MAM] model={model_path}")
    print(
        f"[KDV-MAM] r={num_modes}, q={num_secondary}, p={degree}, "
        f"regularizer=(c={float(model['ridge_c']):.3e}, A={float(model['ridge_a']):.3e}, "
        f"H={float(model['ridge_h']):.3e}, P={float(model['ridge_p']):.3e}), "
        f"rk4_substeps={rk4_substeps}"
    )
    print(f"[KDV-MAM] q0 nonlinear LS cost={float(q0_result.cost):.3e}, nfev={int(q0_result.nfev)}")
    print(f"[KDV-MAM] train error={train_error:.6e}")
    print(f"[KDV-MAM] prediction error={prediction_error:.6e}")
    print(f"[KDV-MAM] full error={full_error:.6e}")
    print(f"[KDV-MAM] summary saved: {summary_path}")
    return rom, full_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAM-OpInf for KdV.")
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "mam_opinf_r5_p2_q9.npz"))
    parser.add_argument("--snapshot-file", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "MAM"))
    parser.add_argument("--max-norm", type=float, default=1e6)
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        snapshot_file=args.snapshot_file,
        results_dir=args.results_dir,
        max_norm=args.max_norm,
    )
