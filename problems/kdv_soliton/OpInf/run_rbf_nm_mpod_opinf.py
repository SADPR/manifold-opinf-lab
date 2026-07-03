#!/usr/bin/env python3
"""Run RBF Nonlinear-Map MPOD-OpInf for the KdV soliton."""

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
from rbf_nm_mpod_opinf_utils import (
    load_fom_dataset,
    load_model,
    plot_error_history,
    plot_snapshot_comparison,
    plot_spacetime_comparison,
    project_snapshots,
    reconstruct_rbf_nm_mpod_snapshots,
    relative_error_history,
    relative_state_error,
    rollout_rk4,
)


def main(
    model_path=os.path.join(SCRIPT_DIR, "models", "rbf_nm_mpod_opinf_r5_q9.npz"),
    snapshot_file=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "RBF-NM-MPOD", "r5_q9"),
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
    weights = np.asarray(model["weights"], dtype=np.float64)
    centers = np.asarray(model["centers"], dtype=np.float64)
    q_mean = np.asarray(model["q_mean"], dtype=np.float64)
    q_scale = np.asarray(model["q_scale"], dtype=np.float64)
    coeffs = np.asarray(model["coeffs"], dtype=np.float64)
    kernel = str(model["kernel"])
    epsilon = float(model["epsilon"])
    num_modes = int(model["num_modes"])
    num_secondary = int(model["num_secondary"])
    train_final_time = float(model["train_final_time"])
    rk4_substeps = int(model.get("rk4_substeps", 1))
    include_quadratic = bool(model.get("include_quadratic", True))
    label = "RBF-NM-MPOD-OpInf" if include_quadratic else "RBF-NM-MPOD no-quadratic"
    file_prefix = "rbf_nm_mpod_opinf" if include_quadratic else "rbf_nm_mpod_noquad_opinf"
    dynamics = "dq/dt = c + A q + H q_quad + P phi_RBF(q)" if include_quadratic else "dq/dt = c + A q + P phi_RBF(q)"

    q_fom = project_snapshots(snapshots, basis, u_ref)
    q_rom, unstable, unstable_index = rollout_rk4(
        q_fom[:, 0],
        times,
        coeffs,
        centers,
        kernel,
        epsilon,
        q_mean,
        q_scale,
        max_norm=max_norm,
        substeps=rk4_substeps,
        include_quadratic=include_quadratic,
    )
    rom = reconstruct_rbf_nm_mpod_snapshots(q_rom, basis, basis_bar, weights, u_ref, centers, kernel, epsilon, q_mean, q_scale)

    prediction_mask = times >= train_final_time - 1e-14
    train_error = relative_state_error(snapshots[:, train_mask], rom[:, train_mask], u_ref=u_ref)
    prediction_error = relative_state_error(snapshots[:, prediction_mask], rom[:, prediction_mask], u_ref=u_ref)
    full_error = relative_state_error(snapshots, rom, u_ref=u_ref)
    error_history = relative_error_history(snapshots, rom)

    tag = f"r{num_modes}_q{num_secondary}"
    spacetime_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_spacetime.png")
    snapshot_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_snapshots.png")
    error_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_error_history.png")
    q_path = os.path.join(results_dir, f"{file_prefix}_{tag}_q_rollout.npz")
    summary_path = os.path.join(results_dir, f"{file_prefix}_{tag}_summary.txt")

    plot_spacetime_comparison(
        x,
        times,
        snapshots,
        rom,
        train_final_time,
        spacetime_plot,
        title=f"KdV {label}, r={num_modes}, q={num_secondary}",
        rom_label=label,
    )
    plot_snapshot_comparison(
        x,
        times,
        snapshots,
        rom,
        snapshot_plot,
        snapshot_times=(train_final_time, times[-1]),
        rom_label=label,
    )
    plot_error_history(
        times,
        error_history,
        train_final_time,
        error_plot,
        title=f"{label} error history, r={num_modes}, q={num_secondary}",
    )
    np.savez(
        q_path,
        times=times,
        q_fom=q_fom,
        q_rom=q_rom,
        error_history=error_history,
        unstable=np.asarray(int(bool(unstable)), dtype=np.int64),
        unstable_index=np.asarray(int(unstable_index), dtype=np.int64),
    )

    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/run_rbf_nm_mpod_opinf.py")]),
            (
                "model",
                [
                    ("model_path", model_path),
                    ("snapshot_file", snapshot_file),
                    ("num_modes_r", num_modes),
                    ("num_secondary_q", num_secondary),
                    ("total_modes_r_plus_q", int(model["total_modes"])),
                    ("kernel", kernel),
                    ("epsilon", epsilon),
                    ("rbf_ridge", model["rbf_ridge"]),
                    ("num_rbf_features", int(model["num_rbf_features"])),
                    ("ridge_c", model["ridge_c"]),
                    ("ridge_a", model["ridge_a"]),
                    ("ridge_h", model["ridge_h"]),
                    ("ridge_rbf", model["ridge_rbf"]),
                    ("include_quadratic", include_quadratic),
                    ("dynamics", dynamics),
                    ("regularizer_convention", model.get("regularizer_convention", "opinf_direct_weight")),
                    ("energy_captured_metric", model["energy_captured"]),
                    ("relative_reconstruction_error", model["relative_reconstruction_error"]),
                    ("relative_derivative_error", model["relative_derivative_error"]),
                    ("training_rollout_error_selected", model["training_rollout_error"]),
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
                    ("rk4_substeps", rk4_substeps),
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
    print("            KDV RBF-NM-MPOD-OPINF ROLLOUT")
    print("====================================================")
    print(f"[KDV-RBF-NM-MPOD] model={model_path}")
    print(f"[KDV-RBF-NM-MPOD] include_quadratic={include_quadratic}")
    if include_quadratic:
        print(
            f"[KDV-RBF-NM-MPOD] r={num_modes}, q={num_secondary}, kernel={kernel}, eps={epsilon:.3e}, "
            f"regularizer=(c={float(model['ridge_c']):.3e}, A={float(model['ridge_a']):.3e}, "
            f"H={float(model['ridge_h']):.3e}, RBF={float(model['ridge_rbf']):.3e})"
        )
    else:
        print(
            f"[KDV-RBF-NM-MPOD] r={num_modes}, q={num_secondary}, kernel={kernel}, eps={epsilon:.3e}, "
            f"regularizer=(c={float(model['ridge_c']):.3e}, A={float(model['ridge_a']):.3e}, "
            f"RBF={float(model['ridge_rbf']):.3e})"
        )
    print(f"[KDV-RBF-NM-MPOD] train error={train_error:.6e}")
    print(f"[KDV-RBF-NM-MPOD] prediction error={prediction_error:.6e}")
    print(f"[KDV-RBF-NM-MPOD] full error={full_error:.6e}")
    print(f"[KDV-RBF-NM-MPOD] summary saved: {summary_path}")
    return rom, full_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RBF-NM-MPOD-OpInf for KdV.")
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "rbf_nm_mpod_opinf_r5_q9.npz"))
    parser.add_argument("--snapshot-file", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "RBF-NM-MPOD", "r5_q9"))
    parser.add_argument("--max-norm", type=float, default=1e6)
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        snapshot_file=args.snapshot_file,
        results_dir=args.results_dir,
        max_norm=args.max_norm,
    )
