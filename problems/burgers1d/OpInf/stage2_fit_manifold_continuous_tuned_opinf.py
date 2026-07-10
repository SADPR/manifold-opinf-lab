#!/usr/bin/env python3
"""Stage 2: tune a continuous-time induced-MPOD OpInf operator.

Mirrors stage2_fit_ann_continuous_tuned_opinf.py so the polynomial-manifold
(MPOD) baseline gets the same ridge x RK4-substep grid search, with the same
train/validation trajectory split and rollout-based selection criterion, as
every ANN-NM-MPOD variant. The earlier MPOD baseline used a single hard-coded
ridge (see stage1_fit_manifold_opinf.py) with no validation search, which is
not a fair comparison against a tuned method.
"""

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
    manifold_continuous_feature_matrix,
    manifold_decode,
    rollout_continuous_rk4,
    save_manifold_model,
)
from opinf_lab.regression import effective_degrees_of_freedom
from opinf_utils import load_pod_data, project_snapshots
from stage1_fit_linear_opinf import write_txt_report


def _default_model_path(num_primary, num_secondary, polynomial_order=2):
    return os.path.join(
        SCRIPT_DIR,
        "models",
        f"mpod_induced_continuous_tuned_r{int(num_primary)}_q{int(num_secondary)}_p{int(polynomial_order)}.npz",
    )


def _energy_captured(sigma, n_keep):
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0 or int(n_keep) > sigma.size:
        return np.nan
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else 1.0


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _trajectory_split(n_items, validation_fraction, random_seed):
    n_items = int(n_items)
    frac = float(validation_fraction)
    all_idx = np.arange(n_items, dtype=np.int64)
    if frac <= 0.0 or n_items < 3:
        return all_idx, np.zeros(0, dtype=np.int64)
    n_val = int(np.floor(frac * n_items))
    n_val = min(max(1, n_val), n_items - 1)
    rng = np.random.default_rng(int(random_seed))
    perm = rng.permutation(n_items)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _candidate_model(base_model, fit, ridge, substeps):
    model = dict(base_model)
    model["operator"] = fit["operator"]
    model["x_mean"] = fit["x_mean"]
    model["x_scale"] = fit["x_scale"]
    model["dynamics_ridge"] = float(ridge)
    model["rk4_substeps"] = int(substeps)
    return model


def _rollout_error(trajectories, indices, model, dt, num_steps, max_norm, substeps):
    err_sq = 0.0
    denom_sq = 0.0
    stable_steps_min = int(num_steps)
    for idx in indices:
        traj = trajectories[int(idx)]
        q_true = traj["q_primary"]
        q_pred, stable_steps, _ = rollout_continuous_rk4(
            q_true[:, 0],
            traj["mu"],
            dt,
            num_steps,
            model,
            max_norm=max_norm,
            substeps=substeps,
        )
        stable_steps_min = min(stable_steps_min, int(stable_steps))
        finite = np.all(np.isfinite(q_pred), axis=0)
        if not np.any(finite):
            return np.inf, stable_steps_min
        err_sq += float(np.linalg.norm(q_true[:, finite] - q_pred[:, finite]) ** 2)
        denom_sq += float(np.linalg.norm(q_true[:, finite]) ** 2)
    return (float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0), stable_steps_min


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_primary=10,
    num_secondary=133,
    polynomial_order=2,
    manifold_ridge=1e-8,
    feature_mode=FEATURE_MODE,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-MPOD-Induced", "ContinuousTuned", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    include_param_linear=True,
    include_quadratic=True,
    include_higher=False,
    max_degree=2,
    ridges=(1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6),
    rk4_substeps=(1, 2, 5, 10),
    validation_fraction=0.25,
    random_seed=42,
    max_norm=1e12,
):
    num_primary = int(num_primary)
    num_secondary = int(num_secondary)
    n_total = num_primary + num_secondary
    if model_path is None:
        model_path = _default_model_path(num_primary, num_secondary, polynomial_order)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print(" STAGE 2: TUNE CONTINUOUS INDUCED-MPOD OPINF ROM")
    print("====================================================")
    print(f"[MPOD-Cont-Tuned] num_primary={num_primary}, num_secondary={num_secondary}, p={int(polynomial_order)}")
    print(f"[MPOD-Cont-Tuned] ridge grid={list(ridges)}")
    print(f"[MPOD-Cont-Tuned] RK4 substeps grid={list(rk4_substeps)}")
    print(f"[MPOD-Cont-Tuned] validation_fraction={float(validation_fraction):.3f}")

    exponents = higher_monomial_exponents(num_primary, polynomial_order)
    print(f"[MPOD-Cont-Tuned] induced higher features={exponents.shape[0]}")

    basis_total, sigma, u_ref, metadata, basis_path, _, _ = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=n_total,
    )
    del metadata
    basis_primary = basis_total[:, :num_primary]
    basis_secondary = basis_total[:, num_primary:n_total]
    mu_samples = get_snapshot_params()
    train_idx, val_idx = _trajectory_split(len(mu_samples), validation_fraction, random_seed)

    trajectories = []
    q_primary_blocks = []
    q_secondary_blocks = []
    t0 = time.time()
    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[MPOD-Cont-Tuned] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
        snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder, verbose=False)
        q_total = project_snapshots(snaps, basis_total, u_ref)
        q_primary = q_total[:num_primary, :]
        q_secondary = q_total[num_primary:n_total, :]
        q_primary_blocks.append(q_primary)
        q_secondary_blocks.append(q_secondary)
        trajectories.append(
            {
                "mu": [float(mu[0]), float(mu[1])],
                "q_primary": q_primary,
                "qdot": estimate_time_derivative(q_primary, dt),
            }
        )
    elapsed_assembly = time.time() - t0

    print(f"[MPOD-Cont-Tuned] train trajectories: {train_idx.tolist()}")
    print(f"[MPOD-Cont-Tuned] validation trajectories: {val_idx.tolist() if val_idx.size else 'none'}")

    # Decoder (Xi) is fit once from all training trajectories, same as the
    # untuned baseline and analogous to how the ANN decoder is trained once
    # and shared across the AL/PQ/QC dynamics-library tiers: it is a
    # closed-form ridge fit (no early-stopping / iterative-overfitting risk),
    # so there is no equivalent benefit to holding out trajectories here.
    q_primary_all = np.hstack(q_primary_blocks)
    q_secondary_all = np.hstack(q_secondary_blocks)
    t0 = time.time()
    xi, manifold_coeff_rel_error = fit_polynomial_manifold(
        q_primary_all, q_secondary_all, polynomial_order=polynomial_order, ridge=manifold_ridge
    )
    elapsed_manifold_fit = time.time() - t0

    base_model = {
        "xi": xi,
        "exponents": exponents,
        "num_primary": num_primary,
        "num_secondary": num_secondary,
        "polynomial_order": int(polynomial_order),
        "manifold_operator_library": "induced_higher",
        "include_manifold_dynamics": True,
        "feature_mode": feature_mode,
        "include_param_linear": bool(include_param_linear),
        "include_quadratic": bool(include_quadratic),
        "include_higher": bool(include_higher),
        "max_degree": int(max_degree),
    }

    def _assemble(indices):
        theta_blocks, qdot_blocks = [], []
        for idx in indices:
            traj = trajectories[int(idx)]
            theta_blocks.append(
                manifold_continuous_feature_matrix(
                    traj["q_primary"],
                    traj["mu"],
                    polynomial_order=polynomial_order,
                    feature_mode=feature_mode,
                    include_param_linear=include_param_linear,
                    include_quadratic=include_quadratic,
                    include_higher=include_higher,
                    max_degree=max_degree,
                    include_manifold_dynamics=True,
                    exponents=exponents,
                )
            )
            qdot_blocks.append(traj["qdot"].T)
        return np.vstack(theta_blocks), np.vstack(qdot_blocks)

    theta_train, qdot_train = _assemble(train_idx)
    theta_all, qdot_all = _assemble(np.arange(len(trajectories)))
    print(f"[MPOD-Cont-Tuned] Train design matrix shape: {theta_train.shape}")

    # The induced higher-order library contains monomials up to degree 2p,
    # which overflow (RuntimeWarning: overflow in power) under weak
    # regularization once a rollout drifts even slightly off the training
    # manifold. A candidate that looks fine on the held-out trajectories
    # (fit on 7/9 trajectories) can still diverge once refit on all 9 -- the
    # two fits see different data and this library is numerically fragile
    # near its stability boundary. So for every grid point we eagerly fit
    # *both* the train-only model (-> genuine held-out validation score) and
    # the all-trajectory refit (-> the model that would actually ship), and
    # select on whichever is worse of the two. This is more expensive than a
    # plain validation sweep but the fits themselves are cheap ridge solves.
    grid_rows = []
    t0 = time.time()
    for i, ridge in enumerate(ridges, start=1):
        print(f"[MPOD-Cont-Tuned][RIDGE] {i}/{len(ridges)} ridge={float(ridge):.4e}", flush=True)
        fit_train = fit_continuous_operator(theta_train, qdot_train, ridge=float(ridge))
        fit_all = fit_continuous_operator(theta_all, qdot_all, ridge=float(ridge))
        for substeps in rk4_substeps:
            if val_idx.size:
                val_candidate = _candidate_model(base_model, fit_train, ridge, substeps)
                val_err, val_stable = _rollout_error(trajectories, val_idx, val_candidate, dt, num_steps, max_norm, substeps)
            else:
                val_err, val_stable = np.nan, int(num_steps)
            all_candidate = _candidate_model(base_model, fit_all, ridge, substeps)
            all_err, all_stable = _rollout_error(
                trajectories, np.arange(len(trajectories)), all_candidate, dt, num_steps, max_norm, substeps
            )
            finite_scores = [e for e in (val_err, all_err) if np.isfinite(e)]
            score = max(val_err, all_err) if len(finite_scores) == 2 else np.inf
            print(
                f"[MPOD-Cont-Tuned][GRID] ridge={float(ridge):.4e}, substeps={int(substeps)}, "
                f"deriv={fit_train['relative_derivative_training_error']:.6e}, "
                f"val_rollout={val_err:.6e}, all_rollout={all_err:.6e}, score={score:.6e}",
                flush=True,
            )
            grid_rows.append(
                {
                    "ridge": float(ridge),
                    "rk4_substeps": int(substeps),
                    "train_derivative_error": float(fit_train["relative_derivative_training_error"]),
                    "validation_rollout_error": float(val_err),
                    "validation_stable_steps": int(val_stable),
                    "all_rollout_error": float(all_err),
                    "all_stable_steps": int(all_stable),
                    "score": float(score),
                    "fit_all": fit_all,
                }
            )
    elapsed_dynamics_fit = time.time() - t0

    finite_rows = [row for row in grid_rows if np.isfinite(row["score"])]
    if not finite_rows:
        raise RuntimeError("All continuous MPOD OpInf candidates failed on either the held-out or full-data rollout.")
    best = min(finite_rows, key=lambda row: row["score"])
    final_fit = best["fit_all"]
    all_rollout_error = best["all_rollout_error"]
    all_stable_steps = best["all_stable_steps"]

    selected_ridge = float(best["ridge"])
    selected_substeps = int(best["rk4_substeps"])
    print(
        f"[MPOD-Cont-Tuned] Selected ridge={selected_ridge:.4e}, rk4_substeps={selected_substeps}, "
        f"val_rollout={best['validation_rollout_error']:.6e}, all_rollout={all_rollout_error:.6e}"
    )
    effective_dof = effective_degrees_of_freedom(theta_all, selected_ridge, penalize_intercept=False)
    print(
        f"[MPOD-Cont-Tuned] Effective DOF={effective_dof:.1f} of {theta_all.shape[1]} nominal features "
        f"({theta_all.shape[0]} training samples)"
    )

    err_sq, denom_sq = 0.0, 0.0
    for mu, q_primary in zip(mu_samples, q_primary_blocks):
        snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder, verbose=False)
        recon = manifold_decode(q_primary, basis_primary, basis_secondary, xi, u_ref, polynomial_order=polynomial_order)
        err_sq += float(np.linalg.norm(snaps - recon) ** 2)
        denom_sq += float(np.linalg.norm(snaps) ** 2)
    manifold_state_rel_error = float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0

    operator = final_fit["operator"]
    save_manifold_model(
        model_path,
        operator=operator,
        x_mean=final_fit["x_mean"],
        x_scale=final_fit["x_scale"],
        xi=xi,
        exponents=exponents,
        num_primary=num_primary,
        num_secondary=num_secondary,
        polynomial_order=int(polynomial_order),
        manifold_feature_type="elementwise_powers",
        manifold_operator_library="induced_higher",
        dynamics_feature_type="parametric_induced_higher_polynomial_mpod_tuned",
        manifold_ridge=float(manifold_ridge),
        dynamics_ridge=selected_ridge,
        rk4_substeps=selected_substeps,
        feature_mode=feature_mode,
        include_param_linear=bool(include_param_linear),
        include_quadratic=bool(include_quadratic),
        include_higher=bool(include_higher),
        include_manifold_dynamics=True,
        max_degree=int(max_degree),
        num_features=int(operator.shape[1]),
        num_higher_features=int(exponents.shape[0]),
        relative_manifold_training_error=manifold_state_rel_error,
        relative_manifold_coeff_training_error=manifold_coeff_rel_error,
        relative_derivative_training_error=float(final_fit["relative_derivative_training_error"]),
        validation_rollout_error=float(best["validation_rollout_error"]),
        training_rollout_error=float(all_rollout_error),
        num_validation_trajectories=int(val_idx.size),
        num_training_trajectories=int(train_idx.size),
        num_training_samples=int(theta_all.shape[0]),
        effective_dof=float(effective_dof),
        ridge_grid=np.asarray(list(ridges), dtype=np.float64),
        rk4_substeps_grid=np.asarray(list(rk4_substeps), dtype=np.int64),
        dt=float(dt),
        num_steps=int(num_steps),
        pod_basis_path=basis_path,
        energy_captured_primary=_energy_captured(sigma, num_primary),
        energy_captured_total_basis=_energy_captured(sigma, n_total),
    )

    summary_path = os.path.join(
        results_dir, f"manifold_continuous_tuned_r{num_primary}_q{num_secondary}_p{int(polynomial_order)}_training_summary.txt"
    )
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage2_fit_manifold_continuous_tuned_opinf.py")]),
            (
                "configuration",
                [
                    ("num_primary", num_primary),
                    ("num_secondary", num_secondary),
                    ("polynomial_order", int(polynomial_order)),
                    ("manifold_operator_library", "induced_higher"),
                    ("manifold_ridge", float(manifold_ridge)),
                    ("ridge_grid", list(ridges)),
                    ("rk4_substeps_grid", list(rk4_substeps)),
                    ("selected_ridge", selected_ridge),
                    ("selected_rk4_substeps", selected_substeps),
                    ("validation_fraction", float(validation_fraction)),
                    ("random_seed", int(random_seed)),
                    ("train_trajectory_indices", train_idx.tolist()),
                    ("validation_trajectory_indices", val_idx.tolist()),
                ],
            ),
            (
                "fit",
                [
                    ("dynamics_higher_feature_dimension", int(exponents.shape[0])),
                    ("dynamics_design_matrix_shape", theta_all.shape),
                    ("operator_shape", operator.shape),
                    ("effective_degrees_of_freedom", float(effective_dof)),
                    ("solve_method", final_fit["solve_method"]),
                    ("relative_manifold_training_error", manifold_state_rel_error),
                    ("relative_manifold_coeff_training_error", manifold_coeff_rel_error),
                    ("relative_derivative_training_error", float(final_fit["relative_derivative_training_error"])),
                    ("selected_validation_rollout_error", float(best["validation_rollout_error"])),
                    ("training_rollout_error", float(all_rollout_error)),
                    ("energy_captured_primary", _energy_captured(sigma, num_primary)),
                    ("energy_captured_total_basis", _energy_captured(sigma, n_total)),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_manifold_fit_seconds", elapsed_manifold_fit),
                    ("elapsed_dynamics_fit_seconds", elapsed_dynamics_fit),
                ],
            ),
            (
                "grid",
                [
                    (f"candidate_{i}", (row["ridge"], row["rk4_substeps"], row["train_derivative_error"], row["validation_rollout_error"], row["score"]))
                    for i, row in enumerate(grid_rows, start=1)
                ],
            ),
            ("outputs", [("model_npz", model_path), ("summary_txt", summary_path)]),
        ],
    )
    print(f"[MPOD-Cont-Tuned] Final derivative training error: {final_fit['relative_derivative_training_error']:.3e}")
    print(f"[MPOD-Cont-Tuned] Training rollout error: {all_rollout_error:.3e}")
    print(f"[MPOD-Cont-Tuned] Saved model: {model_path}")
    print(f"[MPOD-Cont-Tuned] Saved summary: {summary_path}")
    return model_path, final_fit["relative_derivative_training_error"], all_rollout_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune continuous-time induced-MPOD OpInf.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-primary", type=int, default=10)
    parser.add_argument("--num-secondary", type=int, default=133)
    parser.add_argument("--polynomial-order", type=int, default=2)
    parser.add_argument("--manifold-ridge", type=float, default=1e-8)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-MPOD-Induced", "ContinuousTuned", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--ridges", default="1e-4,1e-2,1e0,1e2,1e4,1e6")
    parser.add_argument("--rk4-substeps", default="1,2,5,10")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-norm", type=float, default=1e12)
    args = parser.parse_args()
    main(
        pod_dir=args.pod_dir,
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        polynomial_order=args.polynomial_order,
        manifold_ridge=args.manifold_ridge,
        snap_folder=args.snap_folder,
        model_path=args.model_path,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
        ridges=_parse_float_list(args.ridges),
        rk4_substeps=_parse_int_list(args.rk4_substeps),
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        max_norm=args.max_norm,
    )
