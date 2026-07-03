#!/usr/bin/env python3
"""Tune a continuous-time ANN-NM-MPOD OpInf operator."""

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

from ann_manifold_opinf_utils import (
    ANN_MANIFOLD_MODEL_FAMILY,
    ann_continuous_feature_matrix,
    load_ann_manifold_model,
    rollout_ann_continuous_rk4,
    save_ann_manifold_model,
)
from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps
from manifold_opinf_utils import estimate_time_derivative, fit_continuous_operator
from opinf_utils import load_pod_data, project_snapshots
from stage1_fit_linear_opinf import write_txt_report


DEFAULT_ANN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ann_manifold_linear_plus_ann_r20_q123.npz")


def _default_model_path(num_primary, num_secondary, include_quadratic=False, include_full_manifold_quadratic=False):
    if include_full_manifold_quadratic:
        name = f"ann_nm_mpod_fullquadratic_continuous_tuned_r{int(num_primary)}_q{int(num_secondary)}.npz"
    elif include_quadratic:
        name = f"ann_nm_mpod_quadratic_continuous_tuned_r{int(num_primary)}_q{int(num_secondary)}.npz"
    else:
        name = f"ann_nm_mpod_noquadratic_continuous_tuned_r{int(num_primary)}_q{int(num_secondary)}.npz"
    return os.path.join(
        SCRIPT_DIR,
        "models",
        name,
    )


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _energy_captured(sigma, n_keep):
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0 or int(n_keep) > sigma.size:
        return np.nan
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else 1.0


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


def _assemble_blocks(trajectories, indices, model):
    theta_blocks = []
    qdot_blocks = []
    for idx in indices:
        traj = trajectories[int(idx)]
        theta_blocks.append(
            ann_continuous_feature_matrix(
                traj["q_primary"],
                traj["mu"],
                model,
                include_ann_dynamics=bool(model["include_ann_dynamics"]),
                include_param_ann_dynamics=bool(model["include_param_ann_dynamics"]),
                include_full_manifold_quadratic=bool(model.get("include_full_manifold_quadratic", False)),
            )
        )
        qdot_blocks.append(traj["qdot"].T)
    return np.vstack(theta_blocks), np.vstack(qdot_blocks)


def _rollout_error(trajectories, indices, candidate_model, dt, num_steps, max_norm, substeps):
    err_sq = 0.0
    denom_sq = 0.0
    stable_steps_min = int(num_steps)
    for idx in indices:
        traj = trajectories[int(idx)]
        q_true = traj["q_primary"]
        q_pred, stable_steps, _ = rollout_ann_continuous_rk4(
            q_true[:, 0],
            traj["mu"],
            dt,
            num_steps,
            candidate_model,
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


def _candidate_model(base_model, fit, ridge, substeps):
    model = dict(base_model)
    model["operator"] = fit["operator"]
    model["x_mean"] = fit["x_mean"]
    model["x_scale"] = fit["x_scale"]
    model["dynamics_ridge"] = float(ridge)
    model["rk4_substeps"] = int(substeps)
    model["model_family"] = ANN_MANIFOLD_MODEL_FAMILY
    return model


def main(
    ann_model_path=DEFAULT_ANN_MODEL_PATH,
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "ContinuousTuned", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    ridges=(1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6),
    rk4_substeps=(1, 2, 5, 10),
    validation_fraction=0.25,
    random_seed=42,
    max_norm=1e12,
    device=None,
    include_quadratic=False,
    include_full_manifold_quadratic=False,
):
    os.makedirs(results_dir, exist_ok=True)
    ann_model = load_ann_manifold_model(ann_model_path, device=device)
    num_primary = int(ann_model["num_primary"])
    num_secondary = int(ann_model["num_secondary"])
    n_total = num_primary + num_secondary
    include_full_manifold_quadratic = bool(include_full_manifold_quadratic)
    include_quadratic = bool(include_quadratic) or include_full_manifold_quadratic
    if model_path is None:
        model_path = _default_model_path(
            num_primary,
            num_secondary,
            include_quadratic=include_quadratic,
            include_full_manifold_quadratic=include_full_manifold_quadratic,
        )
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    ann_model["include_quadratic"] = include_quadratic
    ann_model["include_higher"] = False
    ann_model["include_ann_dynamics"] = True
    ann_model["include_full_manifold_quadratic"] = include_full_manifold_quadratic

    print("\n====================================================")
    print("  STAGE 2: TUNE CONTINUOUS ANN-NM-MPOD OPINF ROM")
    print("====================================================")
    print(f"[ANN-Cont-Tuned] source ANN model: {ann_model_path}")
    print(f"[ANN-Cont-Tuned] num_primary={num_primary}, num_secondary={num_secondary}")
    print("[ANN-Cont-Tuned] model form: dq/dt = W theta(q, mu)")
    print(f"[ANN-Cont-Tuned] q quadratic terms: {'enabled' if include_quadratic else 'disabled'}")
    print(
        "[ANN-Cont-Tuned] full manifold quadratic terms: "
        f"{'enabled' if include_full_manifold_quadratic else 'disabled'}"
    )
    print(f"[ANN-Cont-Tuned] ridge grid={list(ridges)}")
    print(f"[ANN-Cont-Tuned] RK4 substeps grid={list(rk4_substeps)}")
    print(f"[ANN-Cont-Tuned] validation_fraction={float(validation_fraction):.3f}")

    basis_total, sigma, u_ref, metadata, basis_path, _, _ = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=n_total,
    )
    del metadata
    mu_samples = get_snapshot_params()
    train_idx, val_idx = _trajectory_split(len(mu_samples), validation_fraction, random_seed)

    trajectories = []
    t0 = time.time()
    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[ANN-Cont-Tuned] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
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
        trajectories.append(
            {
                "mu": [float(mu[0]), float(mu[1])],
                "q_primary": q_primary,
                "qdot": estimate_time_derivative(q_primary, dt),
            }
        )
    elapsed_assembly = time.time() - t0

    print(f"[ANN-Cont-Tuned] train trajectories: {train_idx.tolist()}")
    print(f"[ANN-Cont-Tuned] validation trajectories: {val_idx.tolist() if val_idx.size else 'none'}")
    theta_train, qdot_train = _assemble_blocks(trajectories, train_idx, ann_model)
    print(f"[ANN-Cont-Tuned] Train design matrix shape: {theta_train.shape}")
    print(f"[ANN-Cont-Tuned] Train qdot target shape: {qdot_train.shape}")

    grid_rows = []
    best = None
    fit_by_ridge = {}
    for i, ridge in enumerate(ridges, start=1):
        print(f"[ANN-Cont-Tuned][RIDGE] {i}/{len(ridges)} ridge={float(ridge):.4e}", flush=True)
        fit = fit_continuous_operator(theta_train, qdot_train, ridge=float(ridge))
        fit_by_ridge[float(ridge)] = fit
        for substeps in rk4_substeps:
            candidate = _candidate_model(ann_model, fit, ridge, substeps)
            if val_idx.size:
                validation_rollout_error, validation_stable_steps = _rollout_error(
                    trajectories,
                    val_idx,
                    candidate,
                    dt,
                    num_steps,
                    max_norm=max_norm,
                    substeps=substeps,
                )
                score = validation_rollout_error
            else:
                validation_rollout_error = np.nan
                validation_stable_steps = int(num_steps)
                score = fit["relative_derivative_training_error"]
            print(
                f"[ANN-Cont-Tuned][GRID] ridge={float(ridge):.4e}, "
                f"substeps={int(substeps)}, deriv={fit['relative_derivative_training_error']:.6e}, "
                f"val_rollout={validation_rollout_error:.6e}",
                flush=True,
            )
            row = {
                "ridge": float(ridge),
                "rk4_substeps": int(substeps),
                "train_derivative_error": float(fit["relative_derivative_training_error"]),
                "validation_rollout_error": float(validation_rollout_error),
                "validation_stable_steps": int(validation_stable_steps),
                "score": float(score),
            }
            grid_rows.append(row)
            if np.isfinite(score) and (best is None or score < best["score"]):
                best = {**row, "fit": fit}

    if best is None:
        raise RuntimeError("All continuous ANN OpInf candidates failed.")

    selected_ridge = float(best["ridge"])
    selected_substeps = int(best["rk4_substeps"])
    print(
        f"[ANN-Cont-Tuned] Selected ridge={selected_ridge:.4e}, "
        f"rk4_substeps={selected_substeps}, score={best['score']:.6e}"
    )
    theta_all, qdot_all = _assemble_blocks(trajectories, np.arange(len(trajectories)), ann_model)
    t0 = time.time()
    final_fit = fit_continuous_operator(theta_all, qdot_all, ridge=selected_ridge)
    elapsed_fit = time.time() - t0
    final_model = _candidate_model(ann_model, final_fit, selected_ridge, selected_substeps)
    all_rollout_error, all_stable_steps = _rollout_error(
        trajectories,
        np.arange(len(trajectories)),
        final_model,
        dt,
        num_steps,
        max_norm=max_norm,
        substeps=selected_substeps,
    )

    ann_state_dict = {key: val.detach().cpu().clone() for key, val in ann_model["ann_torch_model"].state_dict().items()}
    save_ann_manifold_model(
        model_path,
        ann_state_dict=ann_state_dict,
        model_family=ANN_MANIFOLD_MODEL_FAMILY,
        operator=final_fit["operator"],
        x_mean=final_fit["x_mean"],
        x_scale=final_fit["x_scale"],
        num_primary=num_primary,
        num_secondary=num_secondary,
        dynamics_feature_type=(
            "continuous_tuned_full_quadratic_manifold"
            if include_full_manifold_quadratic
            else ("continuous_tuned_quadratic_plus_ann" if include_quadratic else "continuous_tuned_linear_plus_ann")
        ),
        dynamics_ridge=selected_ridge,
        rk4_substeps=selected_substeps,
        feature_mode=ann_model["feature_mode"],
        include_param_linear=bool(ann_model["include_param_linear"]),
        include_quadratic=include_quadratic,
        include_higher=False,
        max_degree=int(ann_model["max_degree"]),
        include_ann_dynamics=True,
        include_param_ann_dynamics=bool(ann_model["include_param_ann_dynamics"]),
        include_full_manifold_quadratic=include_full_manifold_quadratic,
        num_features=int(final_fit["operator"].shape[1]),
        ann_include_mu=bool(ann_model["ann_include_mu"]),
        ann_hidden_dims=np.asarray(ann_model["ann_hidden_dims"], dtype=np.int64),
        ann_x_mean=ann_model["ann_x_mean"],
        ann_x_std=ann_model["ann_x_std"],
        ann_y_mean=ann_model["ann_y_mean"],
        ann_y_std=ann_model["ann_y_std"],
        ann_best_val_mse=float(ann_model["ann_best_val_mse"]),
        ann_train_relative_error=float(ann_model["ann_train_relative_error"]),
        ann_validation_relative_error=float(ann_model["ann_validation_relative_error"]),
        ann_full_relative_error=float(ann_model["ann_full_relative_error"]),
        relative_manifold_training_error=float(ann_model["relative_manifold_training_error"]),
        relative_derivative_training_error=float(final_fit["relative_derivative_training_error"]),
        validation_rollout_error=float(best["validation_rollout_error"]),
        training_rollout_error=float(all_rollout_error),
        num_validation_trajectories=int(val_idx.size),
        num_training_trajectories=int(train_idx.size),
        train_trajectory_indices=train_idx,
        validation_trajectory_indices=val_idx,
        ridge_grid=np.asarray(list(ridges), dtype=np.float64),
        rk4_substeps_grid=np.asarray(list(rk4_substeps), dtype=np.int64),
        grid_results=np.asarray(
            [
                [
                    row["ridge"],
                    row["rk4_substeps"],
                    row["train_derivative_error"],
                    row["validation_rollout_error"],
                    row["score"],
                ]
                for row in grid_rows
            ],
            dtype=np.float64,
        ),
        source_ann_model_path=ann_model_path,
        dt=float(dt),
        num_steps=int(num_steps),
        pod_basis_path=basis_path,
        energy_captured_primary=_energy_captured(sigma, num_primary),
        energy_captured_total_basis=_energy_captured(sigma, n_total),
    )

    summary_path = os.path.join(results_dir, f"ann_continuous_tuned_r{num_primary}_q{num_secondary}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage2_fit_ann_continuous_tuned_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", ANN_MANIFOLD_MODEL_FAMILY),
                    ("source_ann_model_path", ann_model_path),
                    ("num_primary", num_primary),
                    ("num_secondary", num_secondary),
                    ("quadratic_terms", include_quadratic),
                    ("full_manifold_quadratic_terms", include_full_manifold_quadratic),
                    ("ridge_grid", list(ridges)),
                    ("rk4_substeps_grid", list(rk4_substeps)),
                    ("selected_ridge", selected_ridge),
                    ("selected_rk4_substeps", selected_substeps),
                    ("validation_fraction", float(validation_fraction)),
                    ("random_seed", int(random_seed)),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("num_cells", int(NUM_CELLS)),
                    ("time_scheme", TIME_SCHEME),
                    ("snap_folder", snap_folder),
                    ("pod_basis_path", basis_path),
                ],
            ),
            (
                "fit",
                [
                    ("train_trajectory_indices", train_idx.tolist()),
                    ("validation_trajectory_indices", val_idx.tolist()),
                    ("train_design_matrix_shape", theta_train.shape),
                    ("all_design_matrix_shape", theta_all.shape),
                    ("operator_shape", final_fit["operator"].shape),
                    ("solve_method", final_fit["solve_method"]),
                    ("relative_derivative_training_error", float(final_fit["relative_derivative_training_error"])),
                    ("selected_validation_rollout_error", float(best["validation_rollout_error"])),
                    ("training_rollout_error", float(all_rollout_error)),
                    ("all_training_stable_steps_min", int(all_stable_steps)),
                    ("energy_captured_primary", _energy_captured(sigma, num_primary)),
                    ("energy_captured_total_basis", _energy_captured(sigma, n_total)),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_fit_seconds", elapsed_fit),
                ],
            ),
            (
                "grid",
                [
                    (
                        f"candidate_{i}",
                        (
                            row["ridge"],
                            row["rk4_substeps"],
                            row["train_derivative_error"],
                            row["validation_rollout_error"],
                            row["score"],
                        ),
                    )
                    for i, row in enumerate(grid_rows, start=1)
                ],
            ),
            ("outputs", [("model_npz", model_path), ("summary_txt", summary_path)]),
        ],
    )

    print(f"[ANN-Cont-Tuned] Final derivative training error: {final_fit['relative_derivative_training_error']:.3e}")
    print(f"[ANN-Cont-Tuned] Training rollout error: {all_rollout_error:.3e}")
    print(f"[ANN-Cont-Tuned] Saved model: {model_path}")
    print(f"[ANN-Cont-Tuned] Saved summary: {summary_path}")
    return model_path, final_fit["relative_derivative_training_error"], all_rollout_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune continuous-time ANN-NM-MPOD OpInf.")
    parser.add_argument("--ann-model-path", default=DEFAULT_ANN_MODEL_PATH)
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "ContinuousTuned", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--ridges", default="1e-4,1e-2,1e0,1e2,1e4,1e6")
    parser.add_argument("--rk4-substeps", default="1,2,5,10")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-norm", type=float, default=1e12)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--with-quadratic",
        action="store_true",
        help="Include compact q_i q_j terms in the reduced ODE.",
    )
    parser.add_argument(
        "--with-full-manifold-quadratic",
        action="store_true",
        help="Include q_i N_j and compact N_i N_j terms in addition to q_i q_j.",
    )
    args = parser.parse_args()
    main(
        ann_model_path=args.ann_model_path,
        pod_dir=args.pod_dir,
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
        device=args.device,
        include_quadratic=args.with_quadratic,
        include_full_manifold_quadratic=args.with_full_manifold_quadratic,
    )
