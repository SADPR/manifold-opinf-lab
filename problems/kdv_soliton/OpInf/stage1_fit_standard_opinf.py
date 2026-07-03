#!/usr/bin/env python3
"""Fit standard linear-subspace quadratic OpInf for the KdV soliton."""

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
from standard_opinf_utils import (
    compact_quadratic_features,
    compute_pod_basis,
    fit_quadratic_continuous_operator,
    fourth_order_time_derivative,
    load_fom_dataset,
    plot_singular_values,
    project_snapshots,
    reconstruct_snapshots,
    relative_state_error,
    rollout_rk4,
    save_model,
)


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def main(
    snapshot_file=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"),
    model_path=os.path.join(SCRIPT_DIR, "models", "standard_quadratic_opinf_r5.npz"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training"),
    num_modes=5,
    regularizer_c_candidates=(0.0, 1e-2, 1e2),
    regularizer_a_candidates=(0.0, 1e-2, 1e0, 1e2),
    regularizer_h_candidates=(1e4, 1e5, 1e6),
    max_norm=1e6,
):
    os.makedirs(results_dir, exist_ok=True)
    x, times, snapshots, train_mask, data = load_fom_dataset(snapshot_file)
    dt = float(np.median(np.diff(times)))
    train_indices = np.flatnonzero(train_mask)
    train_times = times[train_indices]
    train_snapshots = snapshots[:, train_indices]

    print("\n====================================================")
    print("       KDV STANDARD QUADRATIC OPINF TRAINING")
    print("====================================================")
    print(f"[KDV-OpInf] snapshot_file={snapshot_file}")
    print(f"[KDV-OpInf] num_modes={int(num_modes)}")
    print(f"[KDV-OpInf] train snapshots={train_snapshots.shape[1]}, train window=[{train_times[0]:.4f}, {train_times[-1]:.4f}]")

    basis, sigma, u_ref, energy_captured = compute_pod_basis(train_snapshots, num_modes)
    q_train_full = project_snapshots(train_snapshots, basis, u_ref)
    q_fit = q_train_full[:, 2:-2]
    qdot = fourth_order_time_derivative(q_train_full, dt)
    q0 = q_train_full[:, 0]

    best = None
    rows = []
    for regularizer_c in regularizer_c_candidates:
        for regularizer_a in regularizer_a_candidates:
            for regularizer_h in regularizer_h_candidates:
                fit = fit_quadratic_continuous_operator(
                    q_fit,
                    qdot,
                    ridge_c=float(regularizer_c),
                    ridge_a=float(regularizer_a),
                    ridge_h=float(regularizer_h),
                )
                q_rollout, unstable, unstable_index = rollout_rk4(q0, train_times, fit["coeffs"], max_norm=max_norm)
                train_rom = reconstruct_snapshots(q_rollout, basis, u_ref)
                rollout_error = relative_state_error(train_snapshots, train_rom, u_ref=u_ref)
                if unstable:
                    rollout_error = np.inf
                row = {
                    "regularizer_c": float(regularizer_c),
                    "regularizer_a": float(regularizer_a),
                    "regularizer_h": float(regularizer_h),
                    "ridge_c": float(regularizer_c),
                    "ridge_a": float(regularizer_a),
                    "ridge_h": float(regularizer_h),
                    "relative_derivative_error": fit["relative_derivative_error"],
                    "training_rollout_error": float(rollout_error),
                    "unstable": bool(unstable),
                    "unstable_index": int(unstable_index),
                    "rank": int(fit["rank"]),
                }
                rows.append(row)
                print(
                    "[KDV-OpInf][REG] "
                    f"weights=(c={float(regularizer_c):.1e}, A={float(regularizer_a):.1e}, "
                    f"H={float(regularizer_h):.1e}), "
                    f"deriv={fit['relative_derivative_error']:.3e}, "
                    f"train_rollout={float(rollout_error):.3e}, unstable={bool(unstable)}"
                )
                if best is None or row["training_rollout_error"] < best["row"]["training_rollout_error"]:
                    best = {"row": row, "fit": fit}

    if best is None or not np.isfinite(best["row"]["training_rollout_error"]):
        raise RuntimeError("All OpInf ridge candidates were unstable.")

    selected = best["row"]
    selected_fit = best["fit"]
    num_features = 1 + int(num_modes) + compact_quadratic_features(np.zeros(int(num_modes))).size
    save_model(
        model_path,
        snapshot_file=np.asarray(snapshot_file),
        num_modes=np.asarray(int(num_modes), dtype=np.int64),
        num_features=np.asarray(int(num_features), dtype=np.int64),
        basis=basis,
        sigma=sigma,
        u_ref=u_ref,
        coeffs=selected_fit["coeffs"],
        dt=np.asarray(dt, dtype=np.float64),
        train_final_time=np.asarray(train_times[-1], dtype=np.float64),
        ridge_c=np.asarray(float(selected["ridge_c"]), dtype=np.float64),
        ridge_a=np.asarray(float(selected["ridge_a"]), dtype=np.float64),
        ridge_h=np.asarray(float(selected["ridge_h"]), dtype=np.float64),
        regularizer_convention=np.asarray("opinf_direct_weight"),
        energy_captured=np.asarray(float(energy_captured), dtype=np.float64),
        relative_derivative_error=np.asarray(float(selected["relative_derivative_error"]), dtype=np.float64),
        training_rollout_error=np.asarray(float(selected["training_rollout_error"]), dtype=np.float64),
        unstable_index=np.asarray(int(selected["unstable_index"]), dtype=np.int64),
    )

    csv_path = os.path.join(results_dir, f"standard_opinf_r{int(num_modes)}_ridge_grid.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "regularizer_c,regularizer_a,regularizer_h,"
            "relative_derivative_error,training_rollout_error,unstable,unstable_index,rank\n"
        )
        for row in rows:
            file.write(
                f"{row['regularizer_c']:.16e},{row['regularizer_a']:.16e},{row['regularizer_h']:.16e},"
                f"{row['relative_derivative_error']:.16e},"
                f"{row['training_rollout_error']:.16e},{int(row['unstable'])},"
                f"{row['unstable_index']},{row['rank']}\n"
            )

    singular_plot = os.path.join(results_dir, f"standard_opinf_r{int(num_modes)}_singular_values.png")
    plot_singular_values(sigma, num_modes, singular_plot)

    summary_path = os.path.join(results_dir, f"standard_opinf_r{int(num_modes)}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_standard_opinf.py")]),
            (
                "data",
                [
                    ("snapshot_file", snapshot_file),
                    ("snapshot_shape", snapshots.shape),
                    ("training_snapshot_shape", train_snapshots.shape),
                    ("dt", dt),
                    ("train_final_time", train_times[-1]),
                ],
            ),
            (
                "model",
                [
                    ("model_family", "standard linear-subspace quadratic OpInf"),
                    ("num_modes", int(num_modes)),
                    ("num_features", int(num_features)),
                    ("state_approximation", "s ~= s_ref + V q"),
                    ("dynamics", "dq/dt = c + A q + H q_quad"),
                    ("derivative_estimator", "fourth-order centered finite difference on interior samples"),
                    ("regularizer_convention", "opinf direct weights: objective has ||Gamma O^T||_F^2"),
                    ("energy_captured_training_pod", energy_captured),
                ],
            ),
            (
                "selected_regularization",
                [
                    ("regularizer_c", selected["regularizer_c"]),
                    ("regularizer_a", selected["regularizer_a"]),
                    ("regularizer_h", selected["regularizer_h"]),
                    ("relative_derivative_error", selected["relative_derivative_error"]),
                    ("training_rollout_error", selected["training_rollout_error"]),
                    ("unstable", selected["unstable"]),
                ],
            ),
            (
                "outputs",
                [
                    ("model_npz", model_path),
                    ("ridge_grid_csv", csv_path),
                    ("singular_values_png", singular_plot),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )
    print(
        "[KDV-OpInf] selected regularizer weights="
        f"(c={selected['regularizer_c']:.3e}, A={selected['regularizer_a']:.3e}, "
        f"H={selected['regularizer_h']:.3e})"
    )
    print(f"[KDV-OpInf] model saved: {model_path}")
    print(f"[KDV-OpInf] summary saved: {summary_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit standard quadratic OpInf for KdV.")
    parser.add_argument("--snapshot-file", default=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "standard_quadratic_opinf_r5.npz"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training"))
    parser.add_argument("--num-modes", type=int, default=5)
    parser.add_argument("--regularizer-c-candidates", "--ridge-c-candidates", default="0,1e-2,1e2")
    parser.add_argument("--regularizer-a-candidates", "--ridge-a-candidates", default="0,1e-2,1e0,1e2")
    parser.add_argument("--regularizer-h-candidates", "--ridge-h-candidates", default="1e4,1e5,1e6")
    parser.add_argument("--max-norm", type=float, default=1e6)
    args = parser.parse_args()
    main(
        snapshot_file=args.snapshot_file,
        model_path=args.model_path,
        results_dir=args.results_dir,
        num_modes=args.num_modes,
        regularizer_c_candidates=_parse_float_list(args.regularizer_c_candidates),
        regularizer_a_candidates=_parse_float_list(args.regularizer_a_candidates),
        regularizer_h_candidates=_parse_float_list(args.regularizer_h_candidates),
        max_norm=args.max_norm,
    )
