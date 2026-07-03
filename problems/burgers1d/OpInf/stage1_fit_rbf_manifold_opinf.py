#!/usr/bin/env python3
"""Stage 1: fit full-center RBF-manifold continuous-time OpInf ROM."""

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
from burgers.core import get_snapshot_params, load_or_compute_snaps
from manifold_opinf_utils import FEATURE_MODE, estimate_time_derivative, fit_continuous_operator
from opinf_utils import load_pod_data, project_snapshots
from rbf_manifold_opinf_utils import (
    apply_minmax_scaler,
    build_rbf_input_matrix,
    fit_full_rbf_grid_search,
    fit_minmax_scaler,
    predict_rbf_secondary,
    rbf_continuous_feature_matrix,
    rbf_manifold_decode,
    remove_near_duplicates,
    save_rbf_manifold_model,
)
from stage1_fit_linear_opinf import write_txt_report


def _default_model_path(num_primary, num_secondary):
    return os.path.join(SCRIPT_DIR, "models", f"rbf_manifold_linear_plus_rbf_r{int(num_primary)}_q{int(num_secondary)}.npz")


def _energy_captured(sigma, n_keep):
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0 or int(n_keep) > sigma.size:
        return np.nan
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else 1.0


def _parse_float_list(text):
    if text is None:
        return None
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_str_list(text):
    if text is None:
        return None
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _format_bool(value):
    return "true" if bool(value) else "false"


def _model_stub_for_features(
    feature_mode,
    include_param_linear,
    include_quadratic,
    include_higher,
    max_degree,
    rbf_include_mu,
    include_rbf_dynamics,
    include_param_rbf_dynamics,
    rbf_kernel_name,
    rbf_epsilon,
    rbf_centers,
    rbf_weights,
    rbf_x_min,
    rbf_x_span,
):
    return {
        "feature_mode": feature_mode,
        "include_param_linear": bool(include_param_linear),
        "include_quadratic": bool(include_quadratic),
        "include_higher": bool(include_higher),
        "max_degree": int(max_degree),
        "rbf_include_mu": bool(rbf_include_mu),
        "include_rbf_dynamics": bool(include_rbf_dynamics),
        "include_param_rbf_dynamics": bool(include_param_rbf_dynamics),
        "rbf_kernel_name": str(rbf_kernel_name),
        "rbf_epsilon": float(rbf_epsilon),
        "rbf_centers": np.asarray(rbf_centers, dtype=np.float64),
        "rbf_weights": np.asarray(rbf_weights, dtype=np.float64),
        "rbf_x_min": np.asarray(rbf_x_min, dtype=np.float64),
        "rbf_x_span": np.asarray(rbf_x_span, dtype=np.float64),
        "rbf_feature_min": -1.0,
        "rbf_feature_max": 1.0,
    }


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_primary=20,
    num_secondary=40,
    feature_mode=FEATURE_MODE,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-RBF-Manifold", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    rbf_include_mu=True,
    duplicate_tol=0.0,
    kernel_candidates=("imq",),
    epsilon_values=(0.5, 1.5, 4.0),
    rbf_ridge_values=(1e-8, 1e-5),
    cv_folds=2,
    random_seed=42,
    dynamics_ridge=1e2,
    include_param_linear=True,
    include_quadratic=False,
    include_higher=False,
    max_degree=2,
    include_rbf_dynamics=True,
    include_param_rbf_dynamics=False,
):
    num_primary = int(num_primary)
    num_secondary = int(num_secondary)
    n_total = num_primary + num_secondary
    if num_primary < 1:
        raise ValueError("--num-primary must be positive.")
    if num_secondary < 1:
        raise ValueError("--num-secondary must be positive.")
    if model_path is None:
        model_path = _default_model_path(num_primary, num_secondary)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print("   STAGE 1: FIT FULL-RBF MANIFOLD OPINF ROM")
    print("====================================================")
    print(f"[RBF-OpInf] num_primary={num_primary}, num_secondary={num_secondary}")
    print(f"[RBF-OpInf] rbf_include_mu={_format_bool(rbf_include_mu)}, duplicate_tol={float(duplicate_tol):.3e}")
    print(f"[RBF-OpInf] kernels={tuple(kernel_candidates)}")
    print(f"[RBF-OpInf] epsilon_values={list(epsilon_values)}")
    print(f"[RBF-OpInf] rbf_ridge_values={list(rbf_ridge_values)}")
    print(f"[RBF-OpInf] cv_folds={int(cv_folds)}")
    print(
        "[RBF-OpInf] dynamics terms: "
        f"param_linear={_format_bool(include_param_linear)}, "
        f"quadratic={_format_bool(include_quadratic)}, "
        f"higher={_format_bool(include_higher)}, "
        f"rbf={_format_bool(include_rbf_dynamics)}, "
        f"param_rbf={_format_bool(include_param_rbf_dynamics)}, "
        f"ridge={float(dynamics_ridge):.3e}"
    )

    basis_total, sigma, u_ref, metadata, basis_path, _, _ = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=n_total,
    )
    del metadata
    basis_primary = basis_total[:, :num_primary]
    basis_secondary = basis_total[:, num_primary:n_total]
    mu_samples = get_snapshot_params()

    q_primary_blocks = []
    q_secondary_blocks = []
    qdot_blocks = []
    x_blocks = []
    t0 = time.time()

    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[RBF-OpInf] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
        snaps = load_or_compute_snaps(
            mu,
            GRID_X,
            U0,
            dt,
            num_steps,
            snap_folder=snap_folder,
            verbose=False,
        )
        q_total = project_snapshots(snaps, basis_total, u_ref)
        q_primary = q_total[:num_primary, :]
        q_secondary = q_total[num_primary:n_total, :]
        qdot = estimate_time_derivative(q_primary, dt)
        q_primary_blocks.append(q_primary)
        q_secondary_blocks.append(q_secondary)
        qdot_blocks.append(qdot.T)
        x_blocks.append(build_rbf_input_matrix(q_primary, mu, include_mu=rbf_include_mu))

    q_primary_all = np.hstack(q_primary_blocks)
    q_secondary_all = np.hstack(q_secondary_blocks)
    x_all = np.vstack(x_blocks)
    y_all = q_secondary_all.T
    scaler = fit_minmax_scaler(x_all, feature_range=(-1.0, 1.0))
    x_scaled_all = apply_minmax_scaler(x_all, scaler)
    x_filtered, y_filtered, keep_mask = remove_near_duplicates(
        x_scaled_all,
        y_all,
        duplicate_tol=duplicate_tol,
    )
    elapsed_assembly = time.time() - t0

    print(f"[RBF-OpInf] Full RBF samples before filtering: {x_all.shape[0]}")
    print(f"[RBF-OpInf] Full RBF centers after filtering: {x_filtered.shape[0]}")
    print("[RBF-OpInf] Grid-searching full-center RBF decoder")
    t0 = time.time()
    rbf_fit = fit_full_rbf_grid_search(
        x_filtered,
        y_filtered,
        kernel_candidates=kernel_candidates,
        epsilon_values=epsilon_values,
        ridge_values=rbf_ridge_values,
        cv_folds=cv_folds,
        random_seed=random_seed,
        verbose=True,
    )
    elapsed_rbf_fit = time.time() - t0
    print(
        "[RBF-OpInf] Best RBF: "
        f"kernel={rbf_fit['kernel_name']}, eps={rbf_fit['epsilon']:.3e}, "
        f"ridge={rbf_fit['rbf_ridge']:.3e}, "
        f"cv_error={rbf_fit['cv_relative_error']:.3e}"
    )

    model_stub = _model_stub_for_features(
        feature_mode=feature_mode,
        include_param_linear=include_param_linear,
        include_quadratic=include_quadratic,
        include_higher=include_higher,
        max_degree=max_degree,
        rbf_include_mu=rbf_include_mu,
        include_rbf_dynamics=include_rbf_dynamics,
        include_param_rbf_dynamics=include_param_rbf_dynamics,
        rbf_kernel_name=rbf_fit["kernel_name"],
        rbf_epsilon=rbf_fit["epsilon"],
        rbf_centers=rbf_fit["centers"],
        rbf_weights=rbf_fit["weights"],
        rbf_x_min=scaler["x_min"],
        rbf_x_span=scaler["x_span"],
    )

    theta_blocks = []
    for mu, q_primary in zip(mu_samples, q_primary_blocks):
        theta_blocks.append(
            rbf_continuous_feature_matrix(
                q_primary,
                mu,
                model_stub,
                include_rbf_dynamics=include_rbf_dynamics,
                include_param_rbf_dynamics=include_param_rbf_dynamics,
            )
        )
    theta = np.vstack(theta_blocks)
    qdot = np.vstack(qdot_blocks)
    print(f"[RBF-OpInf] Dynamics design matrix shape: {theta.shape}")
    print(f"[RBF-OpInf] Dynamics target matrix shape: {qdot.shape}")
    print("[RBF-OpInf] Fitting continuous-time reduced operators")
    t0 = time.time()
    dyn_fit = fit_continuous_operator(theta, qdot, ridge=dynamics_ridge)
    elapsed_dynamics_fit = time.time() - t0

    # Full-state reconstruction error through the learned RBF manifold on true q.
    err_sq = 0.0
    denom_sq = 0.0
    for mu, q_primary in zip(mu_samples, q_primary_blocks):
        snaps = load_or_compute_snaps(
            mu,
            GRID_X,
            U0,
            dt,
            num_steps,
            snap_folder=snap_folder,
            verbose=False,
        )
        recon = rbf_manifold_decode(
            q_primary,
            basis_primary,
            basis_secondary,
            model_stub,
            u_ref,
            mu,
        )
        err_sq += float(np.linalg.norm(snaps - recon) ** 2)
        denom_sq += float(np.linalg.norm(snaps) ** 2)
    manifold_state_rel_error = float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0

    save_rbf_manifold_model(
        model_path,
        operator=dyn_fit["operator"],
        x_mean=dyn_fit["x_mean"],
        x_scale=dyn_fit["x_scale"],
        num_primary=num_primary,
        num_secondary=num_secondary,
        dynamics_feature_type=(
            "linear_plus_quadratic_plus_rbf"
            if include_quadratic and include_rbf_dynamics
            else "linear_plus_rbf"
            if include_rbf_dynamics
            else "linear_plus_quadratic"
            if include_quadratic
            else "linear"
        ),
        dynamics_ridge=float(dynamics_ridge),
        feature_mode=feature_mode,
        include_param_linear=bool(include_param_linear),
        include_quadratic=bool(include_quadratic),
        include_higher=bool(include_higher),
        max_degree=int(max_degree),
        include_rbf_dynamics=bool(include_rbf_dynamics),
        include_param_rbf_dynamics=bool(include_param_rbf_dynamics),
        num_features=int(dyn_fit["operator"].shape[1]),
        rbf_include_mu=bool(rbf_include_mu),
        rbf_centers=rbf_fit["centers"],
        rbf_weights=rbf_fit["weights"],
        rbf_x_min=scaler["x_min"],
        rbf_x_span=scaler["x_span"],
        rbf_feature_min=float(scaler["feature_min"]),
        rbf_feature_max=float(scaler["feature_max"]),
        rbf_kernel_name=rbf_fit["kernel_name"],
        rbf_epsilon=float(rbf_fit["epsilon"]),
        rbf_ridge=float(rbf_fit["rbf_ridge"]),
        rbf_cv_relative_error=float(rbf_fit["cv_relative_error"]),
        num_rbf_centers=int(rbf_fit["centers"].shape[0]),
        duplicate_tol=float(duplicate_tol),
        relative_manifold_training_error=manifold_state_rel_error,
        relative_derivative_training_error=dyn_fit["relative_derivative_training_error"],
        dt=float(dt),
        num_steps=int(num_steps),
        pod_basis_path=basis_path,
        energy_captured_primary=_energy_captured(sigma, num_primary),
        energy_captured_total_basis=_energy_captured(sigma, n_total),
    )

    summary_path = os.path.join(results_dir, f"rbf_manifold_continuous_r{num_primary}_q{num_secondary}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_rbf_manifold_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", "rbf_manifold_continuous"),
                    ("num_primary", num_primary),
                    ("num_secondary", num_secondary),
                    ("rbf_include_mu", bool(rbf_include_mu)),
                    ("duplicate_tol", float(duplicate_tol)),
                    ("kernel_candidates", list(kernel_candidates)),
                    ("epsilon_values", list(epsilon_values)),
                    ("rbf_ridge_values", list(rbf_ridge_values)),
                    ("cv_folds", int(cv_folds)),
                    ("random_seed", int(random_seed)),
                    ("dynamics_ridge", float(dynamics_ridge)),
                    ("include_param_linear", bool(include_param_linear)),
                    ("include_quadratic", bool(include_quadratic)),
                    ("include_higher", bool(include_higher)),
                    ("max_degree", int(max_degree)),
                    ("include_rbf_dynamics", bool(include_rbf_dynamics)),
                    ("include_param_rbf_dynamics", bool(include_param_rbf_dynamics)),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("num_cells", int(NUM_CELLS)),
                    ("time_scheme", TIME_SCHEME),
                    ("num_training_parameters", len(mu_samples)),
                    ("snap_folder", snap_folder),
                    ("pod_basis_path", basis_path),
                ],
            ),
            (
                "fit",
                [
                    ("num_rbf_samples_before_filtering", int(x_all.shape[0])),
                    ("num_rbf_centers", int(rbf_fit["centers"].shape[0])),
                    ("rbf_input_dimension", int(x_filtered.shape[1])),
                    ("rbf_output_dimension", int(y_filtered.shape[1])),
                    ("best_rbf_kernel", rbf_fit["kernel_name"]),
                    ("best_rbf_epsilon", float(rbf_fit["epsilon"])),
                    ("best_rbf_ridge", float(rbf_fit["rbf_ridge"])),
                    ("rbf_cv_relative_error", float(rbf_fit["cv_relative_error"])),
                    ("rbf_training_relative_error", float(rbf_fit["training_relative_error"])),
                    ("dynamics_design_matrix_shape", theta.shape),
                    ("dynamics_target_matrix_shape", qdot.shape),
                    ("operator_shape", dyn_fit["operator"].shape),
                    ("dynamics_solve_method", dyn_fit["solve_method"]),
                    ("relative_manifold_training_error", manifold_state_rel_error),
                    ("relative_derivative_training_error", dyn_fit["relative_derivative_training_error"]),
                    ("energy_captured_primary", _energy_captured(sigma, num_primary)),
                    ("energy_captured_total_basis", _energy_captured(sigma, n_total)),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_rbf_fit_seconds", elapsed_rbf_fit),
                    ("elapsed_dynamics_fit_seconds", elapsed_dynamics_fit),
                ],
            ),
            (
                "outputs",
                [
                    ("model_npz", model_path),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )

    print(f"[RBF-OpInf] Manifold state training error: {manifold_state_rel_error:.3e}")
    print(f"[RBF-OpInf] Derivative training error: {dyn_fit['relative_derivative_training_error']:.3e}")
    print(f"[RBF-OpInf] Saved model: {model_path}")
    print(f"[RBF-OpInf] Saved summary: {summary_path}")
    return model_path, manifold_state_rel_error, dyn_fit["relative_derivative_training_error"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit full-center RBF-manifold continuous-time OpInf ROM.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-primary", type=int, default=20)
    parser.add_argument("--num-secondary", type=int, default=40)
    parser.add_argument("--feature-mode", default=FEATURE_MODE)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-RBF-Manifold", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--no-rbf-mu", action="store_true")
    parser.add_argument("--duplicate-tol", type=float, default=0.0)
    parser.add_argument("--kernels", default="imq")
    parser.add_argument("--epsilons", default="0.5,1.5,4.0")
    parser.add_argument("--rbf-ridges", default="1e-8,1e-5")
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dynamics-ridge", type=float, default=1e2)
    parser.add_argument("--no-param-linear", action="store_true")
    parser.add_argument("--with-quadratic", action="store_true")
    parser.add_argument("--no-quadratic", action="store_true", help="Deprecated; quadratic dynamics are off by default.")
    parser.add_argument("--with-higher", action="store_true")
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--with-rbf-dynamics", action="store_true", help="Deprecated; RBF dynamics are on by default.")
    parser.add_argument("--no-rbf-dynamics", action="store_true")
    parser.add_argument("--with-param-rbf-dynamics", action="store_true")
    args = parser.parse_args()
    main(
        pod_dir=args.pod_dir,
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        feature_mode=args.feature_mode,
        snap_folder=args.snap_folder,
        model_path=args.model_path,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
        rbf_include_mu=not args.no_rbf_mu,
        duplicate_tol=args.duplicate_tol,
        kernel_candidates=_parse_str_list(args.kernels),
        epsilon_values=_parse_float_list(args.epsilons),
        rbf_ridge_values=_parse_float_list(args.rbf_ridges),
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
        dynamics_ridge=args.dynamics_ridge,
        include_param_linear=not args.no_param_linear,
        include_quadratic=bool(args.with_quadratic and not args.no_quadratic),
        include_higher=bool(args.with_higher),
        max_degree=args.max_degree,
        include_rbf_dynamics=bool((not args.no_rbf_dynamics) or args.with_rbf_dynamics),
        include_param_rbf_dynamics=bool(args.with_param_rbf_dynamics),
    )
