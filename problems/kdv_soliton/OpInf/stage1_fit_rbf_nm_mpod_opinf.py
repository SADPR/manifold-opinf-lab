#!/usr/bin/env python3
"""Fit RBF Nonlinear-Map MPOD-OpInf for the KdV soliton."""

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
    compute_rbf_nm_mpod_manifold,
    fit_rbf_nm_mpod_continuous_operator,
    fourth_order_time_derivative,
    load_fom_dataset,
    reconstruct_rbf_nm_mpod_snapshots,
    relative_state_error,
    rollout_rk4,
    save_model,
)
from standard_opinf_utils import plot_singular_values


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_str_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def main(
    snapshot_file=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"),
    model_path=os.path.join(SCRIPT_DIR, "models", "rbf_nm_mpod_opinf_r5_q9.npz"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training", "rbf_nm_mpod_r5_q9"),
    num_modes=5,
    total_modes=14,
    kernels=("imq", "gaussian"),
    epsilons=(0.25, 0.5),
    rbf_ridges=(1e-8, 1e-6),
    regularizer_ca_candidates=(1e0,),
    regularizer_h_candidates=(1e4,),
    regularizer_rbf_candidates=(1e4, 1e6),
    center_stride=1,
    max_centers=0,
    rk4_substeps=1,
    max_norm=1e6,
):
    x, times, snapshots, train_mask, data = load_fom_dataset(snapshot_file)
    dt = float(np.median(np.diff(times)))
    train_indices = np.flatnonzero(train_mask)
    train_times = times[train_indices]
    train_snapshots = snapshots[:, train_indices]
    r = int(num_modes)
    total = int(total_modes)
    q_secondary = total - r
    include_quadratic = True
    os.makedirs(results_dir, exist_ok=True)
    h_candidates = tuple(regularizer_h_candidates)

    print("\n====================================================")
    print("          KDV RBF-NM-MPOD-OPINF TRAINING")
    print("====================================================")
    print(f"[KDV-RBF-NM-MPOD] snapshot_file={snapshot_file}")
    print(f"[KDV-RBF-NM-MPOD] r={r}, q={q_secondary}, total_modes={total}")
    print(f"[KDV-RBF-NM-MPOD] include_quadratic={include_quadratic}")
    print(f"[KDV-RBF-NM-MPOD] train snapshots={train_snapshots.shape[1]}, train window=[{train_times[0]:.4f}, {train_times[-1]:.4f}]")
    print(f"[KDV-RBF-NM-MPOD] kernels={','.join(kernels)}, epsilons={epsilons}, rbf_ridges={rbf_ridges}")

    best = None
    rows = []
    candidates_total = (
        len(tuple(kernels))
        * len(tuple(epsilons))
        * len(tuple(rbf_ridges))
        * len(tuple(regularizer_ca_candidates))
        * len(h_candidates)
        * len(tuple(regularizer_rbf_candidates))
    )
    candidate_index = 0

    for kernel in kernels:
        for epsilon in epsilons:
            for rbf_ridge in rbf_ridges:
                manifold = compute_rbf_nm_mpod_manifold(
                    train_snapshots,
                    num_modes=r,
                    total_modes=total,
                    kernel=kernel,
                    epsilon=float(epsilon),
                    rbf_ridge=float(rbf_ridge),
                    center_stride=center_stride,
                    max_centers=max_centers,
                )
                basis = manifold["basis"]
                basis_bar = manifold["basis_bar"]
                sigma = manifold["sigma"]
                u_ref = manifold["u_ref"]
                weights = manifold["weights"]
                q_train_full = manifold["q"]
                q_fit = q_train_full[:, 2:-2]
                qdot = fourth_order_time_derivative(q_train_full, dt)
                q0 = q_train_full[:, 0]
                centers = manifold["centers"]
                q_mean = manifold["q_mean"]
                q_scale = manifold["q_scale"]
                num_quadratic_features = r * (r + 1) // 2
                num_features = 1 + r + num_quadratic_features + centers.shape[1]

                print(
                    "[KDV-RBF-NM-MPOD][RBF] "
                    f"kernel={kernel}, eps={float(epsilon):.3e}, ridge={float(rbf_ridge):.1e}, "
                    f"centers={centers.shape[1]}, recon={manifold['relative_reconstruction_error']:.3e}"
                )

                for regularizer_ca in regularizer_ca_candidates:
                    for regularizer_h in h_candidates:
                        for regularizer_rbf in regularizer_rbf_candidates:
                            candidate_index += 1
                            fit = fit_rbf_nm_mpod_continuous_operator(
                                q_fit,
                                qdot,
                                centers,
                                kernel,
                                float(epsilon),
                                q_mean,
                                q_scale,
                                ridge_c=float(regularizer_ca),
                                ridge_a=float(regularizer_ca),
                                ridge_h=float(regularizer_h),
                                ridge_rbf=float(regularizer_rbf),
                                include_quadratic=include_quadratic,
                            )
                            q_rollout, unstable, unstable_index = rollout_rk4(
                                q0,
                                train_times,
                                fit["coeffs"],
                                centers,
                                kernel,
                                float(epsilon),
                                q_mean,
                                q_scale,
                                max_norm=max_norm,
                                substeps=rk4_substeps,
                                include_quadratic=include_quadratic,
                            )
                            train_rom = reconstruct_rbf_nm_mpod_snapshots(
                                q_rollout,
                                basis,
                                basis_bar,
                                weights,
                                u_ref,
                                centers,
                                kernel,
                                float(epsilon),
                                q_mean,
                                q_scale,
                            )
                            rollout_error = relative_state_error(train_snapshots, train_rom, u_ref=u_ref)
                            if unstable:
                                rollout_error = np.inf
                            row = {
                                "kernel": str(kernel),
                                "epsilon": float(epsilon),
                                "rbf_ridge": float(rbf_ridge),
                                "regularizer_ca": float(regularizer_ca),
                                "regularizer_h": float(regularizer_h),
                                "regularizer_rbf": float(regularizer_rbf),
                                "relative_reconstruction_error": float(manifold["relative_reconstruction_error"]),
                                "energy_captured": float(manifold["energy_captured"]),
                                "relative_derivative_error": fit["relative_derivative_error"],
                                "training_rollout_error": float(rollout_error),
                                "unstable": bool(unstable),
                                "unstable_index": int(unstable_index),
                                "rank": int(fit["rank"]),
                                "num_centers": int(centers.shape[1]),
                                "num_quadratic_features": int(num_quadratic_features),
                                "num_features": int(num_features),
                            }
                            rows.append(row)
                            print(
                                "[KDV-RBF-NM-MPOD][REG] "
                                f"{candidate_index}/{candidates_total} "
                                f"ca={float(regularizer_ca):.1e}, h={float(regularizer_h):.1e}, "
                                f"rbf={float(regularizer_rbf):.1e}, "
                                f"deriv={fit['relative_derivative_error']:.3e}, "
                                f"train_rollout={float(rollout_error):.3e}, unstable={bool(unstable)}"
                            )
                            if best is None or row["training_rollout_error"] < best["row"]["training_rollout_error"]:
                                best = {
                                    "row": row,
                                    "fit": fit,
                                    "manifold": manifold,
                                }

    if best is None or not np.isfinite(best["row"]["training_rollout_error"]):
        raise RuntimeError("All RBF-NM-MPOD-OpInf candidates were unstable.")

    selected = best["row"]
    selected_fit = best["fit"]
    manifold = best["manifold"]
    basis = manifold["basis"]
    basis_bar = manifold["basis_bar"]
    weights = manifold["weights"]
    u_ref = manifold["u_ref"]
    centers = manifold["centers"]
    q_mean = manifold["q_mean"]
    q_scale = manifold["q_scale"]
    sigma = manifold["sigma"]

    save_model(
        model_path,
        snapshot_file=np.asarray(snapshot_file),
        model_variant=np.asarray("rbf_nm_mpod_quadratic_plus_rbf_features"),
        include_quadratic=np.asarray(int(include_quadratic), dtype=np.int64),
        num_modes=np.asarray(r, dtype=np.int64),
        total_modes=np.asarray(total, dtype=np.int64),
        num_secondary=np.asarray(q_secondary, dtype=np.int64),
        num_features=np.asarray(int(selected["num_features"]), dtype=np.int64),
        num_quadratic_features=np.asarray(int(selected["num_quadratic_features"]), dtype=np.int64),
        num_rbf_features=np.asarray(int(selected["num_centers"]), dtype=np.int64),
        kernel=np.asarray(str(selected["kernel"])),
        epsilon=np.asarray(float(selected["epsilon"]), dtype=np.float64),
        rbf_ridge=np.asarray(float(selected["rbf_ridge"]), dtype=np.float64),
        basis=basis,
        basis_bar=basis_bar,
        sigma=sigma,
        u_ref=u_ref,
        weights=weights,
        centers=centers,
        q_mean=q_mean,
        q_scale=q_scale,
        center_indices=manifold["center_indices"],
        coeffs=selected_fit["coeffs"],
        dt=np.asarray(dt, dtype=np.float64),
        train_final_time=np.asarray(train_times[-1], dtype=np.float64),
        ridge_c=np.asarray(float(selected["regularizer_ca"]), dtype=np.float64),
        ridge_a=np.asarray(float(selected["regularizer_ca"]), dtype=np.float64),
        ridge_h=np.asarray(float(selected["regularizer_h"]), dtype=np.float64),
        ridge_rbf=np.asarray(float(selected["regularizer_rbf"]), dtype=np.float64),
        rk4_substeps=np.asarray(int(rk4_substeps), dtype=np.int64),
        regularizer_convention=np.asarray("opinf_direct_weight"),
        energy_captured=np.asarray(float(manifold["energy_captured"]), dtype=np.float64),
        relative_reconstruction_error=np.asarray(float(manifold["relative_reconstruction_error"]), dtype=np.float64),
        relative_derivative_error=np.asarray(float(selected["relative_derivative_error"]), dtype=np.float64),
        training_rollout_error=np.asarray(float(selected["training_rollout_error"]), dtype=np.float64),
        unstable_index=np.asarray(int(selected["unstable_index"]), dtype=np.int64),
    )

    tag = f"rbf_nm_mpod_opinf_r{r}_q{q_secondary}"
    csv_path = os.path.join(results_dir, f"{tag}_grid.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "kernel,epsilon,rbf_ridge,regularizer_ca,regularizer_h,regularizer_rbf,"
            "relative_reconstruction_error,energy_captured,relative_derivative_error,"
            "training_rollout_error,unstable,unstable_index,rank,num_centers,num_features\n"
        )
        for row in rows:
            file.write(
                f"{row['kernel']},{row['epsilon']:.16e},{row['rbf_ridge']:.16e},"
                f"{row['regularizer_ca']:.16e},{row['regularizer_h']:.16e},{row['regularizer_rbf']:.16e},"
                f"{row['relative_reconstruction_error']:.16e},{row['energy_captured']:.16e},"
                f"{row['relative_derivative_error']:.16e},{row['training_rollout_error']:.16e},"
                f"{int(row['unstable'])},{row['unstable_index']},{row['rank']},"
                f"{row['num_centers']},{row['num_features']}\n"
            )

    singular_plot = os.path.join(results_dir, f"{tag}_singular_values.png")
    plot_singular_values(sigma, total, singular_plot)

    summary_path = os.path.join(results_dir, f"{tag}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_rbf_nm_mpod_opinf.py")]),
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
                    ("model_family", "RBF Nonlinear-Map MPOD-OpInf"),
                    ("state_approximation", "s ~= s_ref + V q + Vbar W phi_RBF(q)"),
                    ("num_modes_r", r),
                    ("num_secondary_q", q_secondary),
                    ("total_modes_r_plus_q", total),
                    ("kernel", selected["kernel"]),
                    ("epsilon", selected["epsilon"]),
                    ("rbf_ridge", selected["rbf_ridge"]),
                    ("num_centers", selected["num_centers"]),
                    ("relative_reconstruction_error", manifold["relative_reconstruction_error"]),
                    ("energy_captured_metric", manifold["energy_captured"]),
                ],
            ),
            (
                "opinf",
                [
                    ("dynamics", "dq/dt = c + A q + H q_quad + P phi_RBF(q)"),
                    ("num_features", selected["num_features"]),
                    ("include_quadratic", include_quadratic),
                    ("derivative_estimator", "fourth-order centered finite difference on interior samples"),
                    ("regularizer_convention", "opinf direct weights: objective has ||Gamma O^T||_F^2"),
                    ("note", "RBF features are a practical nonlinear-feature analog, not the exact polynomial closure induced in the paper."),
                ],
            ),
            (
                "selected_regularization",
                [
                    ("regularizer_ca_for_c_and_A", selected["regularizer_ca"]),
                    ("regularizer_h", selected["regularizer_h"]),
                    ("regularizer_rbf", selected["regularizer_rbf"]),
                    ("relative_derivative_error", selected["relative_derivative_error"]),
                    ("training_rollout_error", selected["training_rollout_error"]),
                    ("unstable", selected["unstable"]),
                ],
            ),
            (
                "outputs",
                [
                    ("model_npz", model_path),
                    ("grid_csv", csv_path),
                    ("singular_values_png", singular_plot),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )
    print(
        "[KDV-RBF-NM-MPOD] selected="
        f"kernel={selected['kernel']}, eps={selected['epsilon']:.3e}, rbf_ridge={selected['rbf_ridge']:.3e}, "
        f"reg=(ca={selected['regularizer_ca']:.3e}, h={selected['regularizer_h']:.3e}, "
        f"rbf={selected['regularizer_rbf']:.3e})"
    )
    print(f"[KDV-RBF-NM-MPOD] model saved: {model_path}")
    print(f"[KDV-RBF-NM-MPOD] summary saved: {summary_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit RBF-NM-MPOD-OpInf for KdV.")
    parser.add_argument("--snapshot-file", default=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "rbf_nm_mpod_opinf_r5_q9.npz"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training", "rbf_nm_mpod_r5_q9"))
    parser.add_argument("--num-modes", type=int, default=5)
    parser.add_argument("--total-modes", type=int, default=14)
    parser.add_argument("--kernels", default="imq,gaussian")
    parser.add_argument("--epsilons", default="0.25,0.5")
    parser.add_argument("--rbf-ridges", default="1e-8,1e-6")
    parser.add_argument("--regularizer-ca-candidates", "--ridge-ca-candidates", default="1e0")
    parser.add_argument("--regularizer-h-candidates", "--ridge-h-candidates", default="1e4")
    parser.add_argument("--regularizer-rbf-candidates", "--ridge-rbf-candidates", default="1e4,1e6")
    parser.add_argument("--center-stride", type=int, default=1)
    parser.add_argument("--max-centers", type=int, default=0)
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--max-norm", type=float, default=1e6)
    args = parser.parse_args()
    main(
        snapshot_file=args.snapshot_file,
        model_path=args.model_path,
        results_dir=args.results_dir,
        num_modes=args.num_modes,
        total_modes=args.total_modes,
        kernels=_parse_str_list(args.kernels),
        epsilons=_parse_float_list(args.epsilons),
        rbf_ridges=_parse_float_list(args.rbf_ridges),
        regularizer_ca_candidates=_parse_float_list(args.regularizer_ca_candidates),
        regularizer_h_candidates=_parse_float_list(args.regularizer_h_candidates),
        regularizer_rbf_candidates=_parse_float_list(args.regularizer_rbf_candidates),
        center_stride=args.center_stride,
        max_centers=args.max_centers,
        rk4_substeps=args.rk4_substeps,
        max_norm=args.max_norm,
    )
