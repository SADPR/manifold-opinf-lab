#!/usr/bin/env python3
"""Fit a standard linear-subspace continuous-time OpInf ROM."""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LAB_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
for path in (PROJECT_ROOT, LAB_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps
from manifold_opinf_utils import (
    FEATURE_MODE,
    continuous_feature_matrix,
    continuous_feature_vector,
    estimate_time_derivative,
    fit_continuous_operator,
)
from opinf_lab.time_integration import rollout_rk4
from opinf_utils import load_pod_data, project_snapshots
from stage1_fit_linear_opinf import write_txt_report


MODEL_FAMILY = "burgers_standard_linear_subspace_continuous_opinf"


def _default_model_path(num_modes):
    return os.path.join(SCRIPT_DIR, "models", f"standard_continuous_quadratic_r{int(num_modes)}.npz")


def _parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _trajectory_split(n_items, validation_fraction, random_seed):
    all_idx = np.arange(int(n_items), dtype=np.int64)
    if float(validation_fraction) <= 0.0 or int(n_items) < 3:
        return all_idx, np.zeros(0, dtype=np.int64)
    n_val = int(np.floor(float(validation_fraction) * int(n_items)))
    n_val = min(max(1, n_val), int(n_items) - 1)
    rng = np.random.default_rng(int(random_seed))
    perm = rng.permutation(int(n_items))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def _energy_captured(sigma, n_keep):
    sigma = np.asarray(sigma, dtype=np.float64)
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else np.nan


def _rhs(q, mu, model):
    theta = continuous_feature_vector(
        q,
        mu,
        feature_mode=model["feature_mode"],
        include_param_linear=bool(model["include_param_linear"]),
        include_quadratic=bool(model["include_quadratic"]),
        include_higher=bool(model["include_higher"]),
        max_degree=int(model["max_degree"]),
    )
    return np.asarray(model["operator"], dtype=np.float64) @ ((theta - model["x_mean"]) / model["x_scale"])


def _rollout_error(trajectories, indices, model, dt, num_steps, max_norm, substeps):
    err_sq = 0.0
    denom_sq = 0.0
    stable_steps_min = int(num_steps)
    for idx in indices:
        traj = trajectories[int(idx)]
        q_true = traj["q"]
        q_pred, stable_steps, _ = rollout_rk4(
            lambda q: _rhs(q, traj["mu"], model),
            q_true[:, 0],
            dt,
            num_steps,
            max_norm=max_norm,
            substeps=substeps,
        )
        stable_steps_min = min(stable_steps_min, int(stable_steps))
        finite = np.all(np.isfinite(q_pred), axis=0)
        if not np.any(finite):
            return np.inf, stable_steps_min
        err_sq += float(np.linalg.norm(q_true[:, finite] - q_pred[:, finite]) ** 2)
        denom_sq += float(np.linalg.norm(q_true[:, finite]) ** 2)
    return float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0, stable_steps_min


def _candidate_model(base_model, fit, ridge, substeps):
    model = dict(base_model)
    model["operator"] = fit["operator"]
    model["x_mean"] = fit["x_mean"]
    model["x_scale"] = fit["x_scale"]
    model["dynamics_ridge"] = float(ridge)
    model["rk4_substeps"] = int(substeps)
    return model


def _assemble_blocks(trajectories, indices, model):
    theta_blocks = []
    qdot_blocks = []
    for idx in indices:
        traj = trajectories[int(idx)]
        theta_blocks.append(
            continuous_feature_matrix(
                traj["q"],
                traj["mu"],
                feature_mode=model["feature_mode"],
                include_param_linear=bool(model["include_param_linear"]),
                include_quadratic=bool(model["include_quadratic"]),
                include_higher=bool(model["include_higher"]),
                max_degree=int(model["max_degree"]),
            )
        )
        qdot_blocks.append(traj["qdot"].T)
    return np.vstack(theta_blocks), np.vstack(qdot_blocks)


def save_model(model_path, **kwargs):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    arrays = {"model_family": np.asarray(MODEL_FAMILY)}
    for key, value in kwargs.items():
        if isinstance(value, str):
            arrays[key] = np.asarray(value)
        elif isinstance(value, (bool, np.bool_)):
            arrays[key] = np.asarray(bool(value), dtype=np.int64)
        elif isinstance(value, (int, np.integer)):
            arrays[key] = np.asarray(int(value), dtype=np.int64)
        elif isinstance(value, (float, np.floating)):
            arrays[key] = np.asarray(float(value), dtype=np.float64)
        else:
            arrays[key] = np.asarray(value)
    np.savez(model_path, **arrays)


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_modes=20,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-Standard", "Continuous", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_quadratic=True,
    include_higher=False,
    max_degree=2,
    ridges=(1e-4, 1e-2, 1e0, 1e2, 1e4),
    rk4_substeps=(1, 2, 5, 10),
    validation_fraction=0.25,
    random_seed=42,
    max_norm=1e12,
):
    num_modes = int(num_modes)
    if model_path is None:
        model_path = _default_model_path(num_modes)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print("  STAGE 1: FIT STANDARD CONTINUOUS OPINF ROM")
    print("====================================================")
    print(f"[Standard-Cont] num_modes={num_modes}")
    print(f"[Standard-Cont] quadratic terms: {'enabled' if include_quadratic else 'disabled'}")
    print(f"[Standard-Cont] ridge grid={list(ridges)}")
    print(f"[Standard-Cont] RK4 substeps grid={list(rk4_substeps)}")

    basis, sigma, u_ref, metadata, basis_path, _, _ = load_pod_data(pod_dir, U0.size, num_modes=num_modes)
    del metadata
    mu_samples = get_snapshot_params()
    train_idx, val_idx = _trajectory_split(len(mu_samples), validation_fraction, random_seed)

    trajectories = []
    t0 = time.time()
    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[Standard-Cont] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
        snaps = load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder, verbose=False)
        q = project_snapshots(snaps, basis, u_ref)
        trajectories.append({"mu": [float(mu[0]), float(mu[1])], "q": q, "qdot": estimate_time_derivative(q, dt)})
    elapsed_assembly = time.time() - t0

    base_model = {
        "feature_mode": feature_mode,
        "include_param_linear": bool(include_param_linear),
        "include_quadratic": bool(include_quadratic),
        "include_higher": bool(include_higher),
        "max_degree": int(max_degree),
    }
    theta_train, qdot_train = _assemble_blocks(trajectories, train_idx, base_model)
    print(f"[Standard-Cont] Train design matrix shape: {theta_train.shape}")
    print(f"[Standard-Cont] Train qdot target shape: {qdot_train.shape}")

    grid_rows = []
    best = None
    for i, ridge in enumerate(ridges, start=1):
        print(f"[Standard-Cont][RIDGE] {i}/{len(ridges)} ridge={float(ridge):.4e}", flush=True)
        fit = fit_continuous_operator(theta_train, qdot_train, ridge=float(ridge))
        for substeps in rk4_substeps:
            candidate = _candidate_model(base_model, fit, ridge, substeps)
            if val_idx.size:
                val_error, val_stable = _rollout_error(
                    trajectories, val_idx, candidate, dt, num_steps, max_norm=max_norm, substeps=substeps
                )
                score = val_error
            else:
                val_error = np.nan
                val_stable = int(num_steps)
                score = fit["relative_derivative_training_error"]
            print(
                f"[Standard-Cont][GRID] ridge={float(ridge):.4e}, substeps={int(substeps)}, "
                f"deriv={fit['relative_derivative_training_error']:.6e}, val_rollout={val_error:.6e}",
                flush=True,
            )
            row = {
                "ridge": float(ridge),
                "rk4_substeps": int(substeps),
                "train_derivative_error": float(fit["relative_derivative_training_error"]),
                "validation_rollout_error": float(val_error),
                "validation_stable_steps": int(val_stable),
                "score": float(score),
            }
            grid_rows.append(row)
            if np.isfinite(score) and (best is None or score < best["score"]):
                best = {**row, "fit": fit}

    if best is None:
        raise RuntimeError("All standard continuous candidates failed.")

    selected_ridge = float(best["ridge"])
    selected_substeps = int(best["rk4_substeps"])
    print(f"[Standard-Cont] Selected ridge={selected_ridge:.4e}, rk4_substeps={selected_substeps}")
    theta_all, qdot_all = _assemble_blocks(trajectories, np.arange(len(trajectories)), base_model)
    t0 = time.time()
    final_fit = fit_continuous_operator(theta_all, qdot_all, ridge=selected_ridge)
    elapsed_fit = time.time() - t0
    final_model = _candidate_model(base_model, final_fit, selected_ridge, selected_substeps)
    all_rollout_error, all_stable = _rollout_error(
        trajectories, np.arange(len(trajectories)), final_model, dt, num_steps, max_norm=max_norm, substeps=selected_substeps
    )

    save_model(
        model_path,
        operator=final_fit["operator"],
        x_mean=final_fit["x_mean"],
        x_scale=final_fit["x_scale"],
        num_modes=num_modes,
        dynamics_ridge=selected_ridge,
        rk4_substeps=selected_substeps,
        feature_mode=feature_mode,
        include_param_linear=bool(include_param_linear),
        include_quadratic=bool(include_quadratic),
        include_higher=bool(include_higher),
        max_degree=int(max_degree),
        num_features=int(final_fit["operator"].shape[1]),
        relative_derivative_training_error=float(final_fit["relative_derivative_training_error"]),
        validation_rollout_error=float(best["validation_rollout_error"]),
        training_rollout_error=float(all_rollout_error),
        num_validation_trajectories=int(val_idx.size),
        train_trajectory_indices=train_idx,
        validation_trajectory_indices=val_idx,
        ridge_grid=np.asarray(list(ridges), dtype=np.float64),
        rk4_substeps_grid=np.asarray(list(rk4_substeps), dtype=np.int64),
        grid_results=np.asarray(
            [
                [row["ridge"], row["rk4_substeps"], row["train_derivative_error"], row["validation_rollout_error"], row["score"]]
                for row in grid_rows
            ],
            dtype=np.float64,
        ),
        dt=float(dt),
        num_steps=int(num_steps),
        pod_basis_path=basis_path,
        energy_captured=_energy_captured(sigma, num_modes),
    )

    csv_path = os.path.join(results_dir, f"standard_continuous_r{num_modes}_grid.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ridge",
                "rk4_substeps",
                "train_derivative_error",
                "validation_rollout_error",
                "validation_stable_steps",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(grid_rows)

    summary_path = os.path.join(results_dir, f"standard_continuous_r{num_modes}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_standard_continuous_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", MODEL_FAMILY),
                    ("num_modes", num_modes),
                    ("feature_mode", feature_mode),
                    ("include_param_linear", bool(include_param_linear)),
                    ("include_quadratic", bool(include_quadratic)),
                    ("include_higher", bool(include_higher)),
                    ("max_degree", int(max_degree)),
                    ("selected_ridge", selected_ridge),
                    ("selected_rk4_substeps", selected_substeps),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("num_cells", int(NUM_CELLS)),
                    ("time_scheme", TIME_SCHEME),
                    ("pod_basis_path", basis_path),
                ],
            ),
            (
                "fit",
                [
                    ("train_design_matrix_shape", theta_train.shape),
                    ("all_design_matrix_shape", theta_all.shape),
                    ("operator_shape", final_fit["operator"].shape),
                    ("relative_derivative_training_error", final_fit["relative_derivative_training_error"]),
                    ("selected_validation_rollout_error", best["validation_rollout_error"]),
                    ("training_rollout_error", all_rollout_error),
                    ("all_training_stable_steps_min", int(all_stable)),
                    ("energy_captured", _energy_captured(sigma, num_modes)),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_fit_seconds", elapsed_fit),
                ],
            ),
            ("outputs", [("model_npz", model_path), ("grid_csv", csv_path), ("summary_txt", summary_path)]),
        ],
    )

    print(f"[Standard-Cont] Final derivative training error: {final_fit['relative_derivative_training_error']:.3e}")
    print(f"[Standard-Cont] Training rollout error: {all_rollout_error:.3e}")
    print(f"[Standard-Cont] Saved model: {model_path}")
    print(f"[Standard-Cont] Saved summary: {summary_path}")
    return model_path, final_fit["relative_derivative_training_error"], all_rollout_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit standard linear-subspace continuous-time OpInf for 1D Burgers.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-modes", type=int, default=20)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-Standard", "Continuous", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--feature-mode", default=FEATURE_MODE)
    parser.add_argument("--no-param-linear", action="store_true")
    parser.add_argument("--no-quadratic", action="store_true")
    parser.add_argument("--with-higher", action="store_true")
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--ridges", default="1e-4,1e-2,1e0,1e2,1e4")
    parser.add_argument("--rk4-substeps", default="1,2,5,10")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-norm", type=float, default=1e12)
    args = parser.parse_args()
    main(
        pod_dir=args.pod_dir,
        num_modes=args.num_modes,
        snap_folder=args.snap_folder,
        model_path=args.model_path,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
        feature_mode=args.feature_mode,
        include_param_linear=not args.no_param_linear,
        include_quadratic=not args.no_quadratic,
        include_higher=bool(args.with_higher),
        max_degree=args.max_degree,
        ridges=_parse_float_list(args.ridges),
        rk4_substeps=_parse_int_list(args.rk4_substeps),
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        max_norm=args.max_norm,
    )
