#!/usr/bin/env python3
"""Stage 1: fit POD-based quadratic-manifold continuous-time OpInf ROM."""

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
from manifold_opinf_utils import (
    FEATURE_MODE,
    estimate_time_derivative,
    fit_continuous_operator,
    fit_polynomial_manifold,
    higher_monomial_exponents,
    manifold_decode,
    manifold_continuous_feature_matrix,
    save_manifold_model,
)
from opinf_utils import load_pod_data, project_snapshots
from stage1_fit_linear_opinf import write_txt_report


def _default_model_path(num_primary, num_secondary, polynomial_order=2):
    return os.path.join(
        SCRIPT_DIR,
        "models",
        f"mpod_induced_continuous_r{int(num_primary)}_q{int(num_secondary)}_p{int(polynomial_order)}.npz",
    )


def _energy_captured(sigma, n_keep):
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0 or int(n_keep) > sigma.size:
        return np.nan
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else 1.0


def _format_bool(value):
    return "true" if bool(value) else "false"


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_primary=20,
    num_secondary=40,
    polynomial_order=2,
    manifold_ridge=1e-8,
    dynamics_ridge=1e2,
    feature_mode=FEATURE_MODE,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-Manifold", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    include_param_linear=True,
    include_quadratic=True,
    include_higher=False,
    max_degree=2,
    include_manifold_dynamics=True,
):
    num_primary = int(num_primary)
    num_secondary = int(num_secondary)
    n_total = num_primary + num_secondary
    if num_primary < 1:
        raise ValueError("--num-primary must be positive.")
    if num_secondary < 1:
        raise ValueError("--num-secondary must be positive.")
    manifold_operator_library = "induced_higher"
    if model_path is None:
        model_path = _default_model_path(num_primary, num_secondary, polynomial_order)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print("   STAGE 1: FIT POD-MANIFOLD CONTINUOUS OPINF ROM")
    print("====================================================")
    print(f"[MPOD-OpInf] num_primary={num_primary}, num_secondary={num_secondary}")
    print(f"[MPOD-OpInf] polynomial_order={int(polynomial_order)}")
    print(f"[MPOD-OpInf] manifold_ridge={float(manifold_ridge):.3e}")
    print(f"[MPOD-OpInf] dynamics_ridge={float(dynamics_ridge):.3e}")
    print(
        "[MPOD-OpInf] dynamics terms: "
        f"param_linear={_format_bool(include_param_linear)}, "
        f"quadratic={_format_bool(include_quadratic)}, "
        f"higher={_format_bool(include_higher)}, "
        f"manifold_g={_format_bool(include_manifold_dynamics)}, "
        f"max_degree={int(max_degree)}"
    )
    print(f"[MPOD-OpInf] manifold_operator_library={manifold_operator_library}")

    exponents = np.empty((0, num_primary), dtype=np.int64)
    if not include_manifold_dynamics:
        raise ValueError("MPOD training requires the induced higher manifold dynamics library.")
    exponents = higher_monomial_exponents(num_primary, polynomial_order)
    print(f"[MPOD-OpInf] induced higher features={exponents.shape[0]}")

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
    theta_blocks = []
    t0 = time.time()

    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[MPOD-OpInf] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
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
        theta = manifold_continuous_feature_matrix(
            q_primary,
            mu,
            polynomial_order=polynomial_order,
            feature_mode=feature_mode,
            include_param_linear=include_param_linear,
            include_quadratic=include_quadratic,
            include_higher=include_higher,
            max_degree=max_degree,
            include_manifold_dynamics=include_manifold_dynamics,
            exponents=exponents,
        )
        q_primary_blocks.append(q_primary)
        q_secondary_blocks.append(q_secondary)
        qdot_blocks.append(qdot.T)
        theta_blocks.append(theta)

    q_primary_all = np.hstack(q_primary_blocks)
    q_secondary_all = np.hstack(q_secondary_blocks)
    elapsed_assembly = time.time() - t0

    print("[MPOD-OpInf] Fitting polynomial manifold closure")
    t0 = time.time()
    xi, manifold_coeff_rel_error = fit_polynomial_manifold(
        q_primary_all,
        q_secondary_all,
        polynomial_order=polynomial_order,
        ridge=manifold_ridge,
    )
    elapsed_manifold_fit = time.time() - t0

    theta = np.vstack(theta_blocks)
    qdot = np.vstack(qdot_blocks)
    print(f"[MPOD-OpInf] Dynamics design matrix shape: {theta.shape}")
    print(f"[MPOD-OpInf] Dynamics target matrix shape: {qdot.shape}")
    print("[MPOD-OpInf] Fitting continuous-time reduced operators")
    t0 = time.time()
    fit = fit_continuous_operator(theta, qdot, ridge=dynamics_ridge)
    elapsed_dynamics_fit = time.time() - t0

    # Full-state reconstruction error through the learned nonlinear manifold.
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
        recon = manifold_decode(
            q_primary,
            basis_primary,
            basis_secondary,
            xi,
            u_ref,
            polynomial_order=polynomial_order,
        )
        err_sq += float(np.linalg.norm(snaps - recon) ** 2)
        denom_sq += float(np.linalg.norm(snaps) ** 2)
    manifold_state_rel_error = float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0

    operator = fit["operator"]
    save_manifold_model(
        model_path,
        operator=operator,
        x_mean=fit["x_mean"],
        x_scale=fit["x_scale"],
        xi=xi,
        exponents=exponents,
        num_primary=num_primary,
        num_secondary=num_secondary,
        polynomial_order=int(polynomial_order),
        manifold_feature_type="elementwise_powers",
        manifold_operator_library=manifold_operator_library,
        dynamics_feature_type="parametric_induced_higher_polynomial_mpod",
        manifold_ridge=float(manifold_ridge),
        dynamics_ridge=float(dynamics_ridge),
        feature_mode=feature_mode,
        include_param_linear=bool(include_param_linear),
        include_quadratic=bool(include_quadratic),
        include_higher=bool(include_higher),
        include_manifold_dynamics=bool(include_manifold_dynamics),
        max_degree=int(max_degree),
        num_features=int(operator.shape[1]),
        num_higher_features=int(exponents.shape[0]),
        relative_manifold_training_error=manifold_state_rel_error,
        relative_manifold_coeff_training_error=manifold_coeff_rel_error,
        relative_derivative_training_error=fit["relative_derivative_training_error"],
        dt=float(dt),
        num_steps=int(num_steps),
        pod_basis_path=basis_path,
        energy_captured_primary=_energy_captured(sigma, num_primary),
        energy_captured_total_basis=_energy_captured(sigma, n_total),
    )

    summary_path = os.path.join(results_dir, f"manifold_continuous_r{num_primary}_q{num_secondary}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_manifold_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", "mpod_quadratic_manifold_continuous"),
                    ("num_primary", num_primary),
                    ("num_secondary", num_secondary),
                    ("polynomial_order", int(polynomial_order)),
                    ("manifold_feature_type", "elementwise_powers"),
                    ("manifold_operator_library", manifold_operator_library),
                    ("dynamics_feature_type", "parametric_induced_higher_polynomial_mpod"),
                    ("manifold_ridge", float(manifold_ridge)),
                    ("dynamics_ridge", float(dynamics_ridge)),
                    ("feature_mode", feature_mode),
                    ("include_param_linear", bool(include_param_linear)),
                    ("include_quadratic", bool(include_quadratic)),
                    ("include_higher", bool(include_higher)),
                    ("include_manifold_dynamics", bool(include_manifold_dynamics)),
                    ("max_degree", int(max_degree)),
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
                    ("manifold_xi_shape", xi.shape),
                    ("manifold_feature_dimension", int(num_primary) * (int(polynomial_order) - 1)),
                    ("dynamics_higher_feature_dimension", int(exponents.shape[0])),
                    ("dynamics_design_matrix_shape", theta.shape),
                    ("dynamics_target_matrix_shape", qdot.shape),
                    ("operator_shape", operator.shape),
                    ("solve_method", fit["solve_method"]),
                    ("relative_manifold_training_error", manifold_state_rel_error),
                    ("relative_manifold_coeff_training_error", manifold_coeff_rel_error),
                    ("relative_derivative_training_error", fit["relative_derivative_training_error"]),
                    ("energy_captured_primary", _energy_captured(sigma, num_primary)),
                    ("energy_captured_total_basis", _energy_captured(sigma, n_total)),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_manifold_fit_seconds", elapsed_manifold_fit),
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

    print(f"[MPOD-OpInf] Manifold state training error: {manifold_state_rel_error:.3e}")
    print(f"[MPOD-OpInf] Derivative training error: {fit['relative_derivative_training_error']:.3e}")
    print(f"[MPOD-OpInf] Saved model: {model_path}")
    print(f"[MPOD-OpInf] Saved summary: {summary_path}")
    return model_path, manifold_state_rel_error, fit["relative_derivative_training_error"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit POD-based quadratic-manifold continuous-time OpInf ROM.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-primary", type=int, default=20)
    parser.add_argument("--num-secondary", type=int, default=40)
    parser.add_argument("--polynomial-order", type=int, default=2)
    parser.add_argument("--manifold-ridge", type=float, default=1e-8)
    parser.add_argument("--dynamics-ridge", type=float, default=1e2)
    parser.add_argument("--feature-mode", default=FEATURE_MODE)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-Manifold", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--no-param-linear", action="store_true")
    parser.add_argument("--no-quadratic", action="store_true")
    parser.add_argument("--with-higher", action="store_true")
    parser.add_argument("--no-higher", action="store_true", help="Deprecated; higher-order dynamics are off by default.")
    parser.add_argument("--max-degree", type=int, default=2)
    args = parser.parse_args()
    main(
        pod_dir=args.pod_dir,
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        polynomial_order=args.polynomial_order,
        manifold_ridge=args.manifold_ridge,
        dynamics_ridge=args.dynamics_ridge,
        feature_mode=args.feature_mode,
        snap_folder=args.snap_folder,
        model_path=args.model_path,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
        include_param_linear=not args.no_param_linear,
        include_quadratic=not args.no_quadratic,
        include_higher=bool(args.with_higher and not args.no_higher),
        max_degree=args.max_degree,
        include_manifold_dynamics=True,
    )
