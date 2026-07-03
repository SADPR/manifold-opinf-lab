#!/usr/bin/env python3
"""Fit POD-based polynomial-manifold OpInf for the KdV soliton."""

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
from mpod_opinf_utils import (
    compute_mpod_manifold,
    fit_mpod_continuous_operator,
    fourth_order_time_derivative,
    higher_monomial_exponents,
    load_fom_dataset,
    reconstruct_mpod_snapshots,
    relative_state_error,
    rollout_rk4,
    save_model,
)
from standard_opinf_utils import plot_singular_values


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def main(
    snapshot_file=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"),
    model_path=os.path.join(SCRIPT_DIR, "models", "mpod_opinf_r5_p2_q9.npz"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training"),
    num_modes=5,
    total_modes=14,
    degree=2,
    gamma=1e-3,
    regularizer_ca_candidates=(1e0,),
    regularizer_h_candidates=(1e4,),
    regularizer_p_candidates=(1e8, 1e10, 1e12),
    max_norm=1e6,
):
    os.makedirs(results_dir, exist_ok=True)
    x, times, snapshots, train_mask, data = load_fom_dataset(snapshot_file)
    dt = float(np.median(np.diff(times)))
    train_indices = np.flatnonzero(train_mask)
    train_times = times[train_indices]
    train_snapshots = snapshots[:, train_indices]
    r = int(num_modes)
    total = int(total_modes)
    p = int(degree)
    q_secondary = total - r

    print("\n====================================================")
    print("             KDV MPOD-OPINF TRAINING")
    print("====================================================")
    print(f"[KDV-MPOD] snapshot_file={snapshot_file}")
    print(f"[KDV-MPOD] r={r}, q={q_secondary}, total_modes={total}, p={p}, gamma={float(gamma):.3e}")
    print(f"[KDV-MPOD] train snapshots={train_snapshots.shape[1]}, train window=[{train_times[0]:.4f}, {train_times[-1]:.4f}]")

    manifold = compute_mpod_manifold(
        train_snapshots,
        num_modes=r,
        total_modes=total,
        degree=p,
        gamma=float(gamma),
    )
    basis = manifold["basis"]
    basis_bar = manifold["basis_bar"]
    sigma = manifold["sigma"]
    u_ref = manifold["u_ref"]
    xi = manifold["xi"]
    q_train_full = manifold["q"]
    q_fit = q_train_full[:, 2:-2]
    qdot = fourth_order_time_derivative(q_train_full, dt)
    q0 = q_train_full[:, 0]
    exponents = higher_monomial_exponents(r, p)
    num_features = 1 + r + r * (r + 1) // 2 + exponents.shape[0]

    print(f"[KDV-MPOD] manifold reconstruction error={manifold['relative_reconstruction_error']:.3e}")
    print(f"[KDV-MPOD] manifold energy metric={manifold['energy_captured']:.3e}")
    print(f"[KDV-MPOD] OpInf features={num_features} (higher={exponents.shape[0]})")

    best = None
    rows = []
    for regularizer_ca in regularizer_ca_candidates:
        for regularizer_h in regularizer_h_candidates:
            for regularizer_p in regularizer_p_candidates:
                fit = fit_mpod_continuous_operator(
                    q_fit,
                    qdot,
                    exponents,
                    ridge_c=float(regularizer_ca),
                    ridge_a=float(regularizer_ca),
                    ridge_h=float(regularizer_h),
                    ridge_p=float(regularizer_p),
                )
                q_rollout, unstable, unstable_index = rollout_rk4(
                    q0,
                    train_times,
                    fit["coeffs"],
                    exponents,
                    max_norm=max_norm,
                )
                train_rom = reconstruct_mpod_snapshots(q_rollout, basis, basis_bar, xi, u_ref, p)
                rollout_error = relative_state_error(train_snapshots, train_rom, u_ref=u_ref)
                if unstable:
                    rollout_error = np.inf
                row = {
                    "regularizer_ca": float(regularizer_ca),
                    "regularizer_h": float(regularizer_h),
                    "regularizer_p": float(regularizer_p),
                    "ridge_c": float(regularizer_ca),
                    "ridge_a": float(regularizer_ca),
                    "ridge_h": float(regularizer_h),
                    "ridge_p": float(regularizer_p),
                    "relative_derivative_error": fit["relative_derivative_error"],
                    "training_rollout_error": float(rollout_error),
                    "unstable": bool(unstable),
                    "unstable_index": int(unstable_index),
                    "rank": int(fit["rank"]),
                }
                rows.append(row)
                print(
                    "[KDV-MPOD][REG] "
                    f"weights=(ca={float(regularizer_ca):.1e}, h={float(regularizer_h):.1e}, "
                    f"p={float(regularizer_p):.1e}), "
                    f"deriv={fit['relative_derivative_error']:.3e}, "
                    f"train_rollout={float(rollout_error):.3e}, unstable={bool(unstable)}"
                )
                if best is None or row["training_rollout_error"] < best["row"]["training_rollout_error"]:
                    best = {"row": row, "fit": fit}

    if best is None or not np.isfinite(best["row"]["training_rollout_error"]):
        raise RuntimeError("All MPOD-OpInf ridge candidates were unstable.")

    selected = best["row"]
    selected_fit = best["fit"]
    save_model(
        model_path,
        snapshot_file=np.asarray(snapshot_file),
        num_modes=np.asarray(r, dtype=np.int64),
        total_modes=np.asarray(total, dtype=np.int64),
        num_secondary=np.asarray(q_secondary, dtype=np.int64),
        degree=np.asarray(p, dtype=np.int64),
        gamma=np.asarray(float(gamma), dtype=np.float64),
        num_features=np.asarray(int(num_features), dtype=np.int64),
        num_higher_features=np.asarray(int(exponents.shape[0]), dtype=np.int64),
        basis=basis,
        basis_bar=basis_bar,
        sigma=sigma,
        u_ref=u_ref,
        xi=xi,
        exponents=exponents,
        coeffs=selected_fit["coeffs"],
        dt=np.asarray(dt, dtype=np.float64),
        train_final_time=np.asarray(train_times[-1], dtype=np.float64),
        ridge_c=np.asarray(float(selected["ridge_c"]), dtype=np.float64),
        ridge_a=np.asarray(float(selected["ridge_a"]), dtype=np.float64),
        ridge_h=np.asarray(float(selected["ridge_h"]), dtype=np.float64),
        ridge_p=np.asarray(float(selected["ridge_p"]), dtype=np.float64),
        regularizer_convention=np.asarray("opinf_direct_weight"),
        energy_captured=np.asarray(float(manifold["energy_captured"]), dtype=np.float64),
        relative_reconstruction_error=np.asarray(float(manifold["relative_reconstruction_error"]), dtype=np.float64),
        relative_derivative_error=np.asarray(float(selected["relative_derivative_error"]), dtype=np.float64),
        training_rollout_error=np.asarray(float(selected["training_rollout_error"]), dtype=np.float64),
        unstable_index=np.asarray(int(selected["unstable_index"]), dtype=np.int64),
    )

    tag = f"mpod_opinf_r{r}_p{p}_q{q_secondary}"
    csv_path = os.path.join(results_dir, f"{tag}_ridge_grid.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "regularizer_ca,regularizer_h,regularizer_p,"
            "relative_derivative_error,training_rollout_error,unstable,unstable_index,rank\n"
        )
        for row in rows:
            file.write(
                f"{row['regularizer_ca']:.16e},{row['regularizer_h']:.16e},{row['regularizer_p']:.16e},"
                f"{row['relative_derivative_error']:.16e},"
                f"{row['training_rollout_error']:.16e},{int(row['unstable'])},"
                f"{row['unstable_index']},{row['rank']}\n"
            )

    singular_plot = os.path.join(results_dir, f"{tag}_singular_values.png")
    plot_singular_values(sigma, total, singular_plot)

    summary_path = os.path.join(results_dir, f"{tag}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_mpod_opinf.py")]),
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
                "manifold",
                [
                    ("model_family", "POD-based polynomial-manifold OpInf"),
                    ("state_approximation", "s ~= s_ref + V q + Vbar Xi g(q)"),
                    ("num_modes_r", r),
                    ("num_secondary_q", q_secondary),
                    ("total_modes_r_plus_q", total),
                    ("polynomial_degree_p", p),
                    ("gamma", float(gamma)),
                    ("relative_reconstruction_error", manifold["relative_reconstruction_error"]),
                    ("energy_captured_metric", manifold["energy_captured"]),
                ],
            ),
            (
                "opinf",
                [
                    ("dynamics", "dq/dt = c + A q + H q_quad + P ghat(q)"),
                    ("num_features", int(num_features)),
                    ("num_higher_features", int(exponents.shape[0])),
                    ("derivative_estimator", "fourth-order centered finite difference on interior samples"),
                    ("regularizer_convention", "opinf direct weights: objective has ||Gamma O^T||_F^2"),
                ],
            ),
            (
                "selected_regularization",
                [
                    ("regularizer_ca_for_c_and_A", selected["regularizer_ca"]),
                    ("regularizer_h", selected["regularizer_h"]),
                    ("regularizer_p", selected["regularizer_p"]),
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
        "[KDV-MPOD] selected regularizer weights="
        f"(ca={selected['regularizer_ca']:.3e}, h={selected['regularizer_h']:.3e}, "
        f"p={selected['regularizer_p']:.3e})"
    )
    print(f"[KDV-MPOD] model saved: {model_path}")
    print(f"[KDV-MPOD] summary saved: {summary_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit MPOD-OpInf for KdV.")
    parser.add_argument("--snapshot-file", default=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "mpod_opinf_r5_p2_q9.npz"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training"))
    parser.add_argument("--num-modes", type=int, default=5)
    parser.add_argument("--total-modes", type=int, default=14)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--regularizer-ca-candidates", "--ridge-ca-candidates", default="1e0")
    parser.add_argument("--regularizer-h-candidates", "--ridge-h-candidates", default="1e4")
    parser.add_argument("--regularizer-p-candidates", "--ridge-p-candidates", default="1e8,1e10,1e12")
    parser.add_argument("--max-norm", type=float, default=1e6)
    args = parser.parse_args()
    main(
        snapshot_file=args.snapshot_file,
        model_path=args.model_path,
        results_dir=args.results_dir,
        num_modes=args.num_modes,
        total_modes=args.total_modes,
        degree=args.degree,
        gamma=args.gamma,
        regularizer_ca_candidates=_parse_float_list(args.regularizer_ca_candidates),
        regularizer_h_candidates=_parse_float_list(args.regularizer_h_candidates),
        regularizer_p_candidates=_parse_float_list(args.regularizer_p_candidates),
        max_norm=args.max_norm,
    )
