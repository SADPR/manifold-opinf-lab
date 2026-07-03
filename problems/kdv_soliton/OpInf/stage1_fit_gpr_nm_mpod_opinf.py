#!/usr/bin/env python3
"""Fit GPR Nonlinear-Map MPOD-OpInf for the KdV soliton."""

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

from gpr_nm_mpod_opinf_utils import (
    compute_gpr_nm_mpod_manifold,
    fit_gpr_nm_mpod_continuous_operator,
    fourth_order_time_derivative,
    load_fom_dataset,
    reconstruct_gpr_nm_mpod_snapshots,
    relative_state_error,
    rollout_rk4,
    save_model,
)
from kdv.core import write_txt_report
from standard_opinf_utils import plot_singular_values


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_str_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_float_pair(text):
    values = _parse_float_list(text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values.")
    return (values[0], values[1])


def main(
    snapshot_file=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"),
    model_path=os.path.join(SCRIPT_DIR, "models", "gpr_nm_mpod_opinf_r5_q9.npz"),
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training", "gpr_nm_mpod_r5_q9"),
    num_modes=5,
    total_modes=14,
    kernels=("gaussian", "matern32"),
    initial_epsilon=0.5,
    initial_noise=1e-6,
    initial_signal_variance=1.0,
    epsilon_bounds=(1e-3, 10.0),
    noise_bounds=(1e-12, 1e-2),
    signal_variance_bounds=(1e-6, 1e6),
    gpr_optimizer_maxiter=60,
    regularizer_ca_candidates=(1e0,),
    regularizer_h_candidates=(1e4,),
    regularizer_gpr_candidates=(1e0, 1e2, 1e4),
    jitter=1e-12,
    rk4_substeps=1,
    max_norm=1e6,
    include_quadratic=True,
):
    x, times, snapshots, train_mask, data = load_fom_dataset(snapshot_file)
    dt = float(np.median(np.diff(times)))
    train_indices = np.flatnonzero(train_mask)
    train_times = times[train_indices]
    train_snapshots = snapshots[:, train_indices]
    r = int(num_modes)
    total = int(total_modes)
    q_secondary = total - r
    include_quadratic = bool(include_quadratic)
    default_model_path = os.path.join(SCRIPT_DIR, "models", "gpr_nm_mpod_opinf_r5_q9.npz")
    default_results_dir = os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training", "gpr_nm_mpod_r5_q9")
    if not include_quadratic and model_path == default_model_path:
        model_path = os.path.join(SCRIPT_DIR, "models", f"gpr_nm_mpod_noquad_opinf_r{r}_q{q_secondary}.npz")
    if not include_quadratic and results_dir == default_results_dir:
        results_dir = os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training", f"gpr_nm_mpod_noquad_r{r}_q{q_secondary}")
    os.makedirs(results_dir, exist_ok=True)
    h_candidates = tuple(regularizer_h_candidates) if include_quadratic else (0.0,)

    print("\n====================================================")
    print("          KDV GPR-NM-MPOD-OPINF TRAINING")
    print("====================================================")
    print(f"[KDV-GPR-NM-MPOD] snapshot_file={snapshot_file}")
    print(f"[KDV-GPR-NM-MPOD] r={r}, q={q_secondary}, total_modes={total}")
    print(f"[KDV-GPR-NM-MPOD] include_quadratic={include_quadratic}")
    print(f"[KDV-GPR-NM-MPOD] train snapshots={train_snapshots.shape[1]}, train window=[{train_times[0]:.4f}, {train_times[-1]:.4f}]")
    print(f"[KDV-GPR-NM-MPOD] GP kernels={','.join(kernels)}")
    print(
        "[KDV-GPR-NM-MPOD] GP hyperparameters optimized by log marginal likelihood "
        f"from eps={initial_epsilon:.3e}, noise={initial_noise:.3e}, signal_var={initial_signal_variance:.3e}"
    )

    manifold = compute_gpr_nm_mpod_manifold(
        train_snapshots,
        num_modes=r,
        total_modes=total,
        optimize_hyperparameters=True,
        kernels=tuple(kernels),
        initial_epsilon=float(initial_epsilon),
        initial_noise=float(initial_noise),
        initial_signal_variance=float(initial_signal_variance),
        epsilon_bounds=tuple(epsilon_bounds),
        noise_bounds=tuple(noise_bounds),
        signal_variance_bounds=tuple(signal_variance_bounds),
        optimizer_maxiter=int(gpr_optimizer_maxiter),
        jitter=float(jitter),
    )
    basis = manifold["basis"]
    basis_bar = manifold["basis_bar"]
    sigma = manifold["sigma"]
    u_ref = manifold["u_ref"]
    alpha = manifold["alpha"]
    q_train_full = manifold["q"]
    q_fit = q_train_full[:, 2:-2]
    qdot = fourth_order_time_derivative(q_train_full, dt)
    q0 = q_train_full[:, 0]
    centers = manifold["centers"]
    q_mean = manifold["q_mean"]
    q_scale = manifold["q_scale"]
    num_quadratic_features = r * (r + 1) // 2 if include_quadratic else 0
    num_features = 1 + r + num_quadratic_features + q_secondary

    print(
        "[KDV-GPR-NM-MPOD][GP-ML] "
        f"kernel={manifold['kernel']}, eps={manifold['epsilon']:.3e}, "
        f"noise={manifold['noise']:.3e}, signal_var={manifold['signal_variance']:.3e}, "
        f"neg_log_ml={manifold['negative_log_marginal_likelihood']:.3e}, "
        f"recon={manifold['relative_reconstruction_error']:.3e}"
    )

    best = None
    rows = []
    candidates_total = (
        len(tuple(regularizer_ca_candidates))
        * len(h_candidates)
        * len(tuple(regularizer_gpr_candidates))
    )
    candidate_index = 0

    for regularizer_ca in regularizer_ca_candidates:
        for regularizer_h in h_candidates:
            for regularizer_gpr in regularizer_gpr_candidates:
                candidate_index += 1
                fit = fit_gpr_nm_mpod_continuous_operator(
                    q_fit,
                    qdot,
                    centers,
                    alpha,
                    manifold["kernel"],
                    float(manifold["epsilon"]),
                    q_mean,
                    q_scale,
                    signal_variance=float(manifold["signal_variance"]),
                    ridge_c=float(regularizer_ca),
                    ridge_a=float(regularizer_ca),
                    ridge_h=float(regularizer_h),
                    ridge_gpr=float(regularizer_gpr),
                    include_quadratic=include_quadratic,
                )
                q_rollout, unstable, unstable_index = rollout_rk4(
                    q0,
                    train_times,
                    fit["coeffs"],
                    centers,
                    alpha,
                    manifold["kernel"],
                    float(manifold["epsilon"]),
                    q_mean,
                    q_scale,
                    signal_variance=float(manifold["signal_variance"]),
                    max_norm=max_norm,
                    substeps=rk4_substeps,
                    include_quadratic=include_quadratic,
                )
                train_rom = reconstruct_gpr_nm_mpod_snapshots(
                    q_rollout,
                    basis,
                    basis_bar,
                    alpha,
                    u_ref,
                    centers,
                    manifold["kernel"],
                    float(manifold["epsilon"]),
                    q_mean,
                    q_scale,
                    signal_variance=float(manifold["signal_variance"]),
                )
                rollout_error = relative_state_error(train_snapshots, train_rom, u_ref=u_ref)
                if unstable:
                    rollout_error = np.inf
                row = {
                    "kernel": str(manifold["kernel"]),
                    "epsilon": float(manifold["epsilon"]),
                    "noise": float(manifold["noise"]),
                    "signal_variance": float(manifold["signal_variance"]),
                    "negative_log_marginal_likelihood": float(manifold["negative_log_marginal_likelihood"]),
                    "regularizer_ca": float(regularizer_ca),
                    "regularizer_h": float(regularizer_h),
                    "regularizer_gpr": float(regularizer_gpr),
                    "relative_reconstruction_error": float(manifold["relative_reconstruction_error"]),
                    "energy_captured": float(manifold["energy_captured"]),
                    "relative_derivative_error": fit["relative_derivative_error"],
                    "training_rollout_error": float(rollout_error),
                    "unstable": bool(unstable),
                    "unstable_index": int(unstable_index),
                    "rank": int(fit["rank"]),
                    "num_quadratic_features": int(num_quadratic_features),
                    "num_features": int(num_features),
                }
                rows.append(row)
                print(
                    "[KDV-GPR-NM-MPOD][REG] "
                    f"{candidate_index}/{candidates_total} "
                    f"ca={float(regularizer_ca):.1e}, h={float(regularizer_h):.1e}, "
                    f"gpr={float(regularizer_gpr):.1e}, "
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
        raise RuntimeError("All GPR-NM-MPOD-OpInf candidates were unstable.")

    selected = best["row"]
    selected_fit = best["fit"]
    manifold = best["manifold"]
    save_model(
        model_path,
        snapshot_file=np.asarray(snapshot_file),
        model_variant=np.asarray(
            "gpr_nm_mpod_quadratic_plus_gpr_secondary_features"
            if include_quadratic
            else "gpr_nm_mpod_linear_plus_gpr_secondary_features_no_quadratic"
        ),
        include_quadratic=np.asarray(int(include_quadratic), dtype=np.int64),
        num_modes=np.asarray(r, dtype=np.int64),
        total_modes=np.asarray(total, dtype=np.int64),
        num_secondary=np.asarray(q_secondary, dtype=np.int64),
        num_features=np.asarray(int(selected["num_features"]), dtype=np.int64),
        num_quadratic_features=np.asarray(int(selected["num_quadratic_features"]), dtype=np.int64),
        num_gpr_features=np.asarray(int(q_secondary), dtype=np.int64),
        kernel=np.asarray(str(selected["kernel"])),
        epsilon=np.asarray(float(selected["epsilon"]), dtype=np.float64),
        noise=np.asarray(float(selected["noise"]), dtype=np.float64),
        jitter=np.asarray(float(jitter), dtype=np.float64),
        signal_variance=np.asarray(float(selected["signal_variance"]), dtype=np.float64),
        gpr_training_objective=np.asarray("log_marginal_likelihood"),
        negative_log_marginal_likelihood=np.asarray(float(selected["negative_log_marginal_likelihood"]), dtype=np.float64),
        gp_optimizer_success=np.asarray(int(bool(manifold["optimizer_success"])), dtype=np.int64),
        gp_optimizer_message=np.asarray(str(manifold["optimizer_message"])),
        gp_optimizer_nit=np.asarray(int(manifold["optimizer_nit"]), dtype=np.int64),
        gp_optimizer_nfev=np.asarray(int(manifold["optimizer_nfev"]), dtype=np.int64),
        basis=manifold["basis"],
        basis_bar=manifold["basis_bar"],
        sigma=manifold["sigma"],
        u_ref=manifold["u_ref"],
        alpha=manifold["alpha"],
        centers=manifold["centers"],
        q_mean=manifold["q_mean"],
        q_scale=manifold["q_scale"],
        coeffs=selected_fit["coeffs"],
        dt=np.asarray(dt, dtype=np.float64),
        train_final_time=np.asarray(train_times[-1], dtype=np.float64),
        ridge_c=np.asarray(float(selected["regularizer_ca"]), dtype=np.float64),
        ridge_a=np.asarray(float(selected["regularizer_ca"]), dtype=np.float64),
        ridge_h=np.asarray(float(selected["regularizer_h"]), dtype=np.float64),
        ridge_gpr=np.asarray(float(selected["regularizer_gpr"]), dtype=np.float64),
        rk4_substeps=np.asarray(int(rk4_substeps), dtype=np.int64),
        regularizer_convention=np.asarray("opinf_direct_weight"),
        energy_captured=np.asarray(float(manifold["energy_captured"]), dtype=np.float64),
        relative_reconstruction_error=np.asarray(float(manifold["relative_reconstruction_error"]), dtype=np.float64),
        relative_derivative_error=np.asarray(float(selected["relative_derivative_error"]), dtype=np.float64),
        training_rollout_error=np.asarray(float(selected["training_rollout_error"]), dtype=np.float64),
        unstable_index=np.asarray(int(selected["unstable_index"]), dtype=np.int64),
    )

    tag = f"gpr_nm_mpod_opinf_r{r}_q{q_secondary}" if include_quadratic else f"gpr_nm_mpod_noquad_opinf_r{r}_q{q_secondary}"
    csv_path = os.path.join(results_dir, f"{tag}_regularizer_grid.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "kernel,epsilon,noise,signal_variance,negative_log_marginal_likelihood,"
            "regularizer_ca,regularizer_h,regularizer_gpr,"
            "relative_reconstruction_error,energy_captured,relative_derivative_error,"
            "training_rollout_error,unstable,unstable_index,rank,num_features\n"
        )
        for row in rows:
            file.write(
                f"{row['kernel']},{row['epsilon']:.16e},{row['noise']:.16e},"
                f"{row['signal_variance']:.16e},{row['negative_log_marginal_likelihood']:.16e},"
                f"{row['regularizer_ca']:.16e},{row['regularizer_h']:.16e},{row['regularizer_gpr']:.16e},"
                f"{row['relative_reconstruction_error']:.16e},{row['energy_captured']:.16e},"
                f"{row['relative_derivative_error']:.16e},{row['training_rollout_error']:.16e},"
                f"{int(row['unstable'])},{row['unstable_index']},{row['rank']},{row['num_features']}\n"
            )

    singular_plot = os.path.join(results_dir, f"{tag}_singular_values.png")
    plot_singular_values(manifold["sigma"], total, singular_plot)

    summary_path = os.path.join(results_dir, f"{tag}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_gpr_nm_mpod_opinf.py")]),
            (
                "manifold",
                [
                    (
                        "model_family",
                        "GPR Nonlinear-Map MPOD-OpInf"
                        if include_quadratic
                        else "GPR Nonlinear-Map MPOD-OpInf no-quadratic ablation",
                    ),
                    ("state_approximation", "s ~= s_ref + V q + Vbar z_GPR(q)"),
                    ("num_modes_r", r),
                    ("num_secondary_q", q_secondary),
                    ("total_modes_r_plus_q", total),
                    ("kernel", selected["kernel"]),
                    ("epsilon", selected["epsilon"]),
                    ("noise", selected["noise"]),
                    ("signal_variance", selected["signal_variance"]),
                    ("gp_training_objective", "log_marginal_likelihood"),
                    ("negative_log_marginal_likelihood", selected["negative_log_marginal_likelihood"]),
                    ("gp_optimizer_success", manifold["optimizer_success"]),
                    ("gp_optimizer_message", manifold["optimizer_message"]),
                    ("gp_optimizer_nit", manifold["optimizer_nit"]),
                    ("gp_optimizer_nfev", manifold["optimizer_nfev"]),
                    ("relative_reconstruction_error", manifold["relative_reconstruction_error"]),
                    ("energy_captured_metric", manifold["energy_captured"]),
                ],
            ),
            (
                "opinf",
                [
                    (
                        "dynamics",
                        "dq/dt = c + A q + H q_quad + B z_GPR(q)"
                        if include_quadratic
                        else "dq/dt = c + A q + B z_GPR(q)",
                    ),
                    ("num_features", selected["num_features"]),
                    ("include_quadratic", include_quadratic),
                    ("derivative_estimator", "fourth-order centered finite difference on interior samples"),
                ],
            ),
            (
                "selected_regularization",
                [
                    ("regularizer_ca_for_c_and_A", selected["regularizer_ca"]),
                    ("regularizer_h", selected["regularizer_h"]),
                    ("regularizer_gpr", selected["regularizer_gpr"]),
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
        "[KDV-GPR-NM-MPOD] selected="
        f"kernel={selected['kernel']}, eps={selected['epsilon']:.3e}, noise={selected['noise']:.3e}, "
        f"signal_var={selected['signal_variance']:.3e}, neg_log_ml={selected['negative_log_marginal_likelihood']:.3e}, "
        f"reg=(ca={selected['regularizer_ca']:.3e}, h={selected['regularizer_h']:.3e}, "
        f"gpr={selected['regularizer_gpr']:.3e})"
    )
    print(f"[KDV-GPR-NM-MPOD] model saved: {model_path}")
    print(f"[KDV-GPR-NM-MPOD] summary saved: {summary_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit GPR-NM-MPOD-OpInf for KdV.")
    parser.add_argument("--snapshot-file", default=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "gpr_nm_mpod_opinf_r5_q9.npz"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training", "gpr_nm_mpod_r5_q9"))
    parser.add_argument("--num-modes", type=int, default=5)
    parser.add_argument("--total-modes", type=int, default=14)
    parser.add_argument("--kernels", default="gaussian,matern32")
    parser.add_argument("--initial-epsilon", type=float, default=0.5)
    parser.add_argument("--initial-noise", type=float, default=1e-6)
    parser.add_argument("--initial-signal-variance", type=float, default=1.0)
    parser.add_argument("--epsilon-bounds", type=_parse_float_pair, default=(1e-3, 10.0))
    parser.add_argument("--noise-bounds", type=_parse_float_pair, default=(1e-12, 1e-2))
    parser.add_argument("--signal-variance-bounds", type=_parse_float_pair, default=(1e-6, 1e6))
    parser.add_argument("--gpr-optimizer-maxiter", type=int, default=60)
    parser.add_argument("--regularizer-ca-candidates", "--ridge-ca-candidates", default="1e0")
    parser.add_argument("--regularizer-h-candidates", "--ridge-h-candidates", default="1e4")
    parser.add_argument("--regularizer-gpr-candidates", "--ridge-gpr-candidates", default="1e0,1e2,1e4")
    parser.add_argument("--jitter", type=float, default=1e-12)
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--max-norm", type=float, default=1e6)
    parser.add_argument(
        "--no-quadratic",
        action="store_true",
        help="Ablation: fit dq/dt = c + A q + B z_GPR(q), omitting H q_quad.",
    )
    args = parser.parse_args()
    main(
        snapshot_file=args.snapshot_file,
        model_path=args.model_path,
        results_dir=args.results_dir,
        num_modes=args.num_modes,
        total_modes=args.total_modes,
        kernels=_parse_str_list(args.kernels),
        initial_epsilon=args.initial_epsilon,
        initial_noise=args.initial_noise,
        initial_signal_variance=args.initial_signal_variance,
        epsilon_bounds=args.epsilon_bounds,
        noise_bounds=args.noise_bounds,
        signal_variance_bounds=args.signal_variance_bounds,
        gpr_optimizer_maxiter=args.gpr_optimizer_maxiter,
        regularizer_ca_candidates=_parse_float_list(args.regularizer_ca_candidates),
        regularizer_h_candidates=_parse_float_list(args.regularizer_h_candidates),
        regularizer_gpr_candidates=_parse_float_list(args.regularizer_gpr_candidates),
        jitter=args.jitter,
        rk4_substeps=args.rk4_substeps,
        max_norm=args.max_norm,
        include_quadratic=not args.no_quadratic,
    )
