#!/usr/bin/env python3
"""Fit a discrete-time ANN-manifold OpInf operator with no quadratic terms."""

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
    ANN_MANIFOLD_DISCRETE_MODEL_FAMILY,
    ann_continuous_feature_matrix,
    load_ann_manifold_model,
    rollout_ann_discrete,
    save_ann_manifold_model,
)
from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps
from opinf_utils import fit_linear_discrete_operator, load_pod_data, project_snapshots
from stage1_fit_linear_opinf import write_txt_report


DEFAULT_ANN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ann_manifold_linear_plus_ann_r20_q123.npz")


def _default_model_path(num_primary, num_secondary):
    return os.path.join(
        SCRIPT_DIR,
        "models",
        f"ann_manifold_discrete_delta_r{int(num_primary)}_q{int(num_secondary)}.npz",
    )


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


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
    delta_blocks = []
    q_current_blocks = []
    q_next_blocks = []
    for idx in indices:
        traj = trajectories[int(idx)]
        q = traj["q_primary"]
        mu = traj["mu"]
        theta_blocks.append(
            ann_continuous_feature_matrix(
                q[:, :-1],
                mu,
                model,
                include_ann_dynamics=bool(model["include_ann_dynamics"]),
                include_param_ann_dynamics=bool(model["include_param_ann_dynamics"]),
            )
        )
        q_current_blocks.append(q[:, :-1].T)
        q_next_blocks.append(q[:, 1:].T)
        delta_blocks.append((q[:, 1:] - q[:, :-1]).T)
    theta = np.vstack(theta_blocks)
    delta = np.vstack(delta_blocks)
    q_current = np.vstack(q_current_blocks)
    q_next = np.vstack(q_next_blocks)
    return theta, delta, q_current, q_next


def _relative_error(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.all(np.isfinite(prediction), axis=0)
    if prediction.ndim == 2 and prediction.shape[0] != reference.shape[0] and prediction.shape[1] == reference.shape[0]:
        finite = np.all(np.isfinite(prediction.T), axis=0)
    if reference.shape != prediction.shape:
        reference = reference.T
    if reference.shape != prediction.shape:
        raise ValueError(f"Shape mismatch in relative error: {reference.shape} vs {prediction.shape}.")
    if reference.ndim == 1:
        finite = np.isfinite(prediction)
    else:
        finite = np.all(np.isfinite(prediction), axis=0)
    if not np.any(finite):
        return np.inf
    ref_eval = reference[..., finite] if reference.ndim > 1 else reference[finite]
    pred_eval = prediction[..., finite] if prediction.ndim > 1 else prediction[finite]
    denom = np.linalg.norm(ref_eval)
    return float(np.linalg.norm(ref_eval - pred_eval) / (denom if denom > 0.0 else 1.0))


def _rollout_error(trajectories, indices, candidate_model, num_steps, max_norm):
    err_sq = 0.0
    denom_sq = 0.0
    stable_steps_min = int(num_steps)
    for idx in indices:
        traj = trajectories[int(idx)]
        q_true = traj["q_primary"]
        q_pred, stable_steps, _ = rollout_ann_discrete(
            q_true[:, 0],
            traj["mu"],
            num_steps,
            candidate_model,
            max_norm=max_norm,
        )
        stable_steps_min = min(stable_steps_min, int(stable_steps))
        finite = np.all(np.isfinite(q_pred), axis=0)
        if not np.any(finite):
            return np.inf, stable_steps_min
        err_sq += float(np.linalg.norm(q_true[:, finite] - q_pred[:, finite]) ** 2)
        denom_sq += float(np.linalg.norm(q_true[:, finite]) ** 2)
    return (float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0), stable_steps_min


def _candidate_model(base_model, fit, ridge):
    model = dict(base_model)
    model["operator"] = fit["operator"]
    model["x_mean"] = fit["x_mean"]
    model["x_scale"] = fit["x_scale"]
    model["dynamics_ridge"] = float(ridge)
    model["discrete_time_map"] = "delta"
    model["model_family"] = ANN_MANIFOLD_DISCRETE_MODEL_FAMILY
    return model


def main(
    ann_model_path=DEFAULT_ANN_MODEL_PATH,
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Discrete", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    ridges=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2),
    validation_fraction=0.25,
    random_seed=42,
    max_norm=1e12,
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    ann_model = load_ann_manifold_model(ann_model_path, device=device)
    num_primary = int(ann_model["num_primary"])
    num_secondary = int(ann_model["num_secondary"])
    n_total = num_primary + num_secondary
    if model_path is None:
        model_path = _default_model_path(num_primary, num_secondary)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    if bool(ann_model.get("include_quadratic", False)):
        raise ValueError("The source ANN model includes quadratic dynamics; this discrete test expects no quadratic terms.")
    ann_model["include_quadratic"] = False
    ann_model["include_higher"] = False
    ann_model["include_ann_dynamics"] = True

    print("\n====================================================")
    print("    STAGE 2: FIT DISCRETE ANN-MANIFOLD OPINF ROM")
    print("====================================================")
    print(f"[ANN-Discrete] source ANN model: {ann_model_path}")
    print(f"[ANN-Discrete] num_primary={num_primary}, num_secondary={num_secondary}")
    print("[ANN-Discrete] model form: q_{k+1} = q_k + W theta(q_k, mu)")
    print("[ANN-Discrete] quadratic terms: disabled")
    print(f"[ANN-Discrete] ridge grid={list(ridges)}")
    print(f"[ANN-Discrete] validation_fraction={float(validation_fraction):.3f}")

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
        print(f"[ANN-Discrete] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
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
        trajectories.append(
            {
                "mu": [float(mu[0]), float(mu[1])],
                "q_primary": q_total[:num_primary, :],
            }
        )
    elapsed_assembly = time.time() - t0

    print(f"[ANN-Discrete] train trajectories: {train_idx.tolist()}")
    print(f"[ANN-Discrete] validation trajectories: {val_idx.tolist() if val_idx.size else 'none'}")
    theta_train, delta_train, q_curr_train, q_next_train = _assemble_blocks(trajectories, train_idx, ann_model)
    print(f"[ANN-Discrete] Train design matrix shape: {theta_train.shape}")
    print(f"[ANN-Discrete] Train delta target shape: {delta_train.shape}")

    grid_rows = []
    best = None
    for i, ridge in enumerate(ridges, start=1):
        print(f"[ANN-Discrete][RIDGE] {i}/{len(ridges)} ridge={float(ridge):.4e}", flush=True)
        fit = fit_linear_discrete_operator(theta_train, delta_train, ridge=float(ridge))
        theta_scaled = (theta_train - fit["x_mean"][None, :]) / fit["x_scale"][None, :]
        delta_pred = theta_scaled @ fit["operator"].T
        q_next_pred = q_curr_train + delta_pred
        train_delta_error = fit["relative_one_step_error"]
        train_q_error = _relative_error(q_next_train.T, q_next_pred.T)
        candidate = _candidate_model(ann_model, fit, ridge)
        if val_idx.size:
            validation_rollout_error, validation_stable_steps = _rollout_error(
                trajectories,
                val_idx,
                candidate,
                num_steps,
                max_norm=max_norm,
            )
            score = validation_rollout_error
        else:
            validation_rollout_error = np.nan
            validation_stable_steps = int(num_steps)
            score = train_q_error
        print(
            f"[ANN-Discrete][RIDGE] ridge={float(ridge):.4e}, "
            f"train_delta={train_delta_error:.6e}, train_q={train_q_error:.6e}, "
            f"val_rollout={validation_rollout_error:.6e}",
            flush=True,
        )
        row = {
            "ridge": float(ridge),
            "train_delta_error": float(train_delta_error),
            "train_q_error": float(train_q_error),
            "validation_rollout_error": float(validation_rollout_error),
            "validation_stable_steps": int(validation_stable_steps),
            "score": float(score),
        }
        grid_rows.append(row)
        if np.isfinite(score) and (best is None or score < best["score"]):
            best = {**row, "fit": fit}

    if best is None:
        raise RuntimeError("All discrete ANN OpInf ridge candidates failed.")

    selected_ridge = float(best["ridge"])
    print(f"[ANN-Discrete] Selected ridge={selected_ridge:.4e}, score={best['score']:.6e}")
    theta_all, delta_all, q_curr_all, q_next_all = _assemble_blocks(trajectories, np.arange(len(trajectories)), ann_model)
    t0 = time.time()
    final_fit = fit_linear_discrete_operator(theta_all, delta_all, ridge=selected_ridge)
    elapsed_fit = time.time() - t0
    theta_scaled_all = (theta_all - final_fit["x_mean"][None, :]) / final_fit["x_scale"][None, :]
    delta_pred_all = theta_scaled_all @ final_fit["operator"].T
    q_next_pred_all = q_curr_all + delta_pred_all
    train_delta_error_all = final_fit["relative_one_step_error"]
    train_q_error_all = _relative_error(q_next_all.T, q_next_pred_all.T)
    final_model = _candidate_model(ann_model, final_fit, selected_ridge)
    all_rollout_error, all_stable_steps = _rollout_error(
        trajectories,
        np.arange(len(trajectories)),
        final_model,
        num_steps,
        max_norm=max_norm,
    )

    ann_state_dict = {key: val.detach().cpu().clone() for key, val in ann_model["ann_torch_model"].state_dict().items()}
    save_ann_manifold_model(
        model_path,
        ann_state_dict=ann_state_dict,
        model_family=ANN_MANIFOLD_DISCRETE_MODEL_FAMILY,
        operator=final_fit["operator"],
        x_mean=final_fit["x_mean"],
        x_scale=final_fit["x_scale"],
        num_primary=num_primary,
        num_secondary=num_secondary,
        dynamics_feature_type="discrete_delta_linear_plus_ann",
        discrete_time_map="delta",
        dynamics_ridge=selected_ridge,
        feature_mode=ann_model["feature_mode"],
        include_param_linear=bool(ann_model["include_param_linear"]),
        include_quadratic=False,
        include_higher=False,
        max_degree=int(ann_model["max_degree"]),
        include_ann_dynamics=True,
        include_param_ann_dynamics=bool(ann_model["include_param_ann_dynamics"]),
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
        relative_one_step_delta_training_error=float(train_delta_error_all),
        relative_one_step_q_training_error=float(train_q_error_all),
        validation_rollout_error=float(best["validation_rollout_error"]),
        training_rollout_error=float(all_rollout_error),
        num_validation_trajectories=int(val_idx.size),
        num_training_trajectories=int(train_idx.size),
        train_trajectory_indices=train_idx,
        validation_trajectory_indices=val_idx,
        ridge_grid=np.asarray(list(ridges), dtype=np.float64),
        ridge_grid_results=np.asarray(
            [
                [
                    row["ridge"],
                    row["train_delta_error"],
                    row["train_q_error"],
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

    summary_path = os.path.join(results_dir, f"ann_discrete_delta_r{num_primary}_q{num_secondary}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage2_fit_ann_discrete_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", ANN_MANIFOLD_DISCRETE_MODEL_FAMILY),
                    ("source_ann_model_path", ann_model_path),
                    ("num_primary", num_primary),
                    ("num_secondary", num_secondary),
                    ("discrete_time_map", "delta"),
                    ("quadratic_terms", False),
                    ("ridge_grid", list(ridges)),
                    ("selected_ridge", selected_ridge),
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
                    ("relative_one_step_delta_training_error", float(train_delta_error_all)),
                    ("relative_one_step_q_training_error", float(train_q_error_all)),
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
                "ridge_grid",
                [
                    (
                        f"candidate_{i}",
                        (
                            row["ridge"],
                            row["train_delta_error"],
                            row["train_q_error"],
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

    print(f"[ANN-Discrete] Final one-step delta training error: {train_delta_error_all:.3e}")
    print(f"[ANN-Discrete] Final one-step q training error: {train_q_error_all:.3e}")
    print(f"[ANN-Discrete] Training rollout error: {all_rollout_error:.3e}")
    print(f"[ANN-Discrete] Saved model: {model_path}")
    print(f"[ANN-Discrete] Saved summary: {summary_path}")
    return model_path, train_q_error_all, all_rollout_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit discrete-time ANN-manifold OpInf without quadratic terms.")
    parser.add_argument("--ann-model-path", default=DEFAULT_ANN_MODEL_PATH)
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Discrete", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--ridges", default="1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-norm", type=float, default=1e12)
    parser.add_argument("--device", default=None)
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
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        max_norm=args.max_norm,
        device=args.device,
    )
