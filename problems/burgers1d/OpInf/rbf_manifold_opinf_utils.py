"""Utilities for full-center RBF nonlinear-manifold OpInf ROMs."""

import os

import numpy as np

from manifold_opinf_utils import (
    continuous_feature_matrix,
    continuous_feature_vector,
    fit_continuous_operator,
)
from opinf_utils import FEATURE_MODE


RBF_MANIFOLD_MODEL_FAMILY = "rbf_manifold_continuous"


def _as_2d_columns(q_columns):
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim == 1:
        q_columns = q_columns.reshape(-1, 1)
    if q_columns.ndim != 2:
        raise ValueError(f"Expected 2D column array, got shape {q_columns.shape}.")
    return q_columns


def build_rbf_input_matrix(q_columns, mu, include_mu=True):
    """Build RBF inputs as rows from primary coordinates and optional parameters."""
    q_columns = _as_2d_columns(q_columns)
    x = q_columns.T
    if include_mu:
        mu = np.asarray(mu, dtype=np.float64).reshape(1, -1)
        if mu.shape[1] != 2:
            raise ValueError(f"Expected two parameters, got shape {mu.shape}.")
        mu_block = np.repeat(mu, x.shape[0], axis=0)
        x = np.hstack((x, mu_block))
    return x


def fit_minmax_scaler(x, feature_range=(-1.0, 1.0)):
    x = np.asarray(x, dtype=np.float64)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    span = x_max - x_min
    span = np.where(span > 1e-14, span, 1.0)
    lo, hi = float(feature_range[0]), float(feature_range[1])
    return {
        "x_min": x_min,
        "x_span": span,
        "feature_min": lo,
        "feature_max": hi,
    }


def apply_minmax_scaler(x, scaler):
    x = np.asarray(x, dtype=np.float64)
    lo = float(scaler["feature_min"])
    hi = float(scaler["feature_max"])
    x_scaled = (x - scaler["x_min"][None, :]) / scaler["x_span"][None, :]
    return lo + (hi - lo) * x_scaled


def remove_near_duplicates(x, y, duplicate_tol=0.0):
    """
    Remove near-duplicate RBF inputs.

    The default tolerance is zero, which keeps all data. Positive tolerances are
    intentionally simple and deterministic; they are for cleanup, not clustering.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tol = float(duplicate_tol)
    if tol <= 0.0:
        return x, y, np.ones(x.shape[0], dtype=bool)

    keep = np.ones(x.shape[0], dtype=bool)
    tol_sq = tol * tol
    for i in range(x.shape[0]):
        if not keep[i]:
            continue
        diff = x[i + 1 :] - x[i]
        close = np.einsum("ij,ij->i", diff, diff) < tol_sq
        if np.any(close):
            idx = np.flatnonzero(close) + i + 1
            keep[idx] = False
    return x[keep], y[keep], keep


def squared_pairwise_distances(a, b=None):
    a = np.asarray(a, dtype=np.float64)
    if b is None:
        b = a
    else:
        b = np.asarray(b, dtype=np.float64)
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def rbf_kernel_from_sqdist(sqdist, epsilon, kernel_name):
    eps = float(epsilon)
    name = str(kernel_name).strip().lower()
    if eps <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if name == "gaussian":
        return np.exp(-(eps * eps) * sqdist)
    if name == "imq":
        return 1.0 / np.sqrt(1.0 + (eps * eps) * sqdist)
    if name == "multiquadric":
        return np.sqrt(1.0 + (eps * eps) * sqdist)
    if name == "matern":
        r = np.sqrt(sqdist)
        z = np.sqrt(3.0) * eps * r
        return (1.0 + z) * np.exp(-z)
    raise ValueError(f"Unsupported RBF kernel {kernel_name!r}.")


def _fit_rbf_weights(x_train, y_train, kernel_name, epsilon, ridge):
    d2 = squared_pairwise_distances(x_train)
    phi = rbf_kernel_from_sqdist(d2, epsilon, kernel_name)
    reg = float(ridge)
    if reg > 0.0:
        phi = phi + reg * np.eye(phi.shape[0], dtype=np.float64)
    try:
        weights = np.linalg.solve(phi, y_train)
        solve_method = "solve"
    except np.linalg.LinAlgError:
        weights, *_ = np.linalg.lstsq(phi, y_train, rcond=None)
        solve_method = "lstsq"
    return weights, solve_method


def predict_rbf_from_scaled_inputs(x_scaled, centers, weights, kernel_name, epsilon):
    d2 = squared_pairwise_distances(x_scaled, centers)
    phi = rbf_kernel_from_sqdist(d2, epsilon, kernel_name)
    return phi @ weights


def kfold_indices(n_samples, n_folds=2, random_seed=42):
    n_samples = int(n_samples)
    n_folds = max(2, min(int(n_folds), n_samples))
    rng = np.random.default_rng(int(random_seed))
    perm = rng.permutation(n_samples)
    folds = np.array_split(perm, n_folds)
    out = []
    all_idx = np.arange(n_samples)
    for val_idx in folds:
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_idx] = True
        out.append((all_idx[~val_mask], all_idx[val_mask]))
    return out


def fit_full_rbf_grid_search(
    x,
    y,
    kernel_candidates=("imq",),
    epsilon_values=(0.5, 1.5, 4.0),
    ridge_values=(1e-8, 1e-5),
    cv_folds=2,
    random_seed=42,
    verbose=False,
    progress_prefix="[RBF-OpInf][GRID]",
):
    """Grid-search full-center RBF hyperparameters and refit on all data."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: x {x.shape}, y {y.shape}.")

    kernels = tuple(str(name).strip().lower() for name in kernel_candidates)
    eps_grid = np.unique(np.asarray(epsilon_values, dtype=np.float64))
    ridge_grid = np.unique(np.asarray(ridge_values, dtype=np.float64))
    if np.any(eps_grid <= 0.0) or np.any(ridge_grid < 0.0):
        raise ValueError("epsilon values must be positive and ridge values nonnegative.")

    folds = kfold_indices(x.shape[0], n_folds=cv_folds, random_seed=random_seed)
    results = []
    total_combinations = len(kernels) * eps_grid.size * ridge_grid.size
    combination_index = 0
    for kernel_name in kernels:
        for epsilon in eps_grid:
            for ridge in ridge_grid:
                combination_index += 1
                if verbose:
                    print(
                        f"{progress_prefix} {combination_index}/{total_combinations} "
                        f"kernel={kernel_name}, eps={float(epsilon):.4e}, "
                        f"ridge={float(ridge):.4e}, folds={len(folds)}",
                        flush=True,
                    )
                fold_errors = []
                failed = False
                for train_idx, val_idx in folds:
                    weights, _ = _fit_rbf_weights(
                        x[train_idx],
                        y[train_idx],
                        kernel_name=kernel_name,
                        epsilon=float(epsilon),
                        ridge=float(ridge),
                    )
                    pred = predict_rbf_from_scaled_inputs(
                        x[val_idx],
                        x[train_idx],
                        weights,
                        kernel_name=kernel_name,
                        epsilon=float(epsilon),
                    )
                    denom = np.linalg.norm(y[val_idx])
                    if denom <= 0.0 or not np.all(np.isfinite(pred)):
                        failed = True
                        break
                    fold_errors.append(float(np.linalg.norm(y[val_idx] - pred) / denom))
                cv_error = np.inf if failed or not fold_errors else float(np.mean(fold_errors))
                if verbose:
                    if np.isfinite(cv_error):
                        print(
                            f"{progress_prefix} {combination_index}/{total_combinations} "
                            f"done cv_relative_error={cv_error:.6e}",
                            flush=True,
                        )
                    else:
                        print(
                            f"{progress_prefix} {combination_index}/{total_combinations} "
                            "failed cv_relative_error=inf",
                            flush=True,
                        )
                results.append(
                    {
                        "kernel": kernel_name,
                        "epsilon": float(epsilon),
                        "ridge": float(ridge),
                        "cv_relative_error": cv_error,
                    }
                )

    valid = [row for row in results if np.isfinite(row["cv_relative_error"])]
    if not valid:
        raise RuntimeError("All RBF grid-search combinations failed.")
    valid.sort(key=lambda row: row["cv_relative_error"])
    best = valid[0]

    if verbose:
        print(
            f"{progress_prefix} best kernel={best['kernel']}, "
            f"eps={best['epsilon']:.4e}, ridge={best['ridge']:.4e}, "
            f"cv_relative_error={best['cv_relative_error']:.6e}; "
            f"refitting on all {x.shape[0]} centers",
            flush=True,
        )
    weights, solve_method = _fit_rbf_weights(
        x,
        y,
        kernel_name=best["kernel"],
        epsilon=best["epsilon"],
        ridge=best["ridge"],
    )
    pred = predict_rbf_from_scaled_inputs(
        x,
        x,
        weights,
        kernel_name=best["kernel"],
        epsilon=best["epsilon"],
    )
    denom = np.linalg.norm(y)
    train_error = float(np.linalg.norm(y - pred) / (denom if denom > 0.0 else 1.0))

    if verbose:
        print(
            f"{progress_prefix} final training_relative_error={train_error:.6e}, "
            f"solve_method={solve_method}",
            flush=True,
        )

    return {
        "centers": x,
        "weights": weights,
        "kernel_name": best["kernel"],
        "epsilon": best["epsilon"],
        "rbf_ridge": best["ridge"],
        "cv_relative_error": best["cv_relative_error"],
        "training_relative_error": train_error,
        "grid_results": results,
        "solve_method": solve_method,
    }


def predict_rbf_secondary(q_primary, mu, model):
    q_primary = _as_2d_columns(q_primary)
    x = build_rbf_input_matrix(
        q_primary,
        mu,
        include_mu=bool(model["rbf_include_mu"]),
    )
    scaler = {
        "x_min": model["rbf_x_min"],
        "x_span": model["rbf_x_span"],
        "feature_min": model["rbf_feature_min"],
        "feature_max": model["rbf_feature_max"],
    }
    x_scaled = apply_minmax_scaler(x, scaler)
    pred = predict_rbf_from_scaled_inputs(
        x_scaled,
        model["rbf_centers"],
        model["rbf_weights"],
        kernel_name=model["rbf_kernel_name"],
        epsilon=model["rbf_epsilon"],
    )
    return pred.T


def rbf_manifold_decode(q_primary, basis_primary, basis_secondary, model, u_ref, mu):
    q_primary = _as_2d_columns(q_primary)
    q_secondary = predict_rbf_secondary(q_primary, mu, model)
    return u_ref[:, None] + basis_primary @ q_primary + basis_secondary @ q_secondary


def rbf_continuous_feature_matrix(
    q_columns,
    mu,
    model,
    include_rbf_dynamics=False,
    include_param_rbf_dynamics=False,
):
    base = continuous_feature_matrix(
        q_columns,
        mu,
        feature_mode=model["feature_mode"],
        include_param_linear=bool(model["include_param_linear"]),
        include_quadratic=bool(model["include_quadratic"]),
        include_higher=bool(model["include_higher"]),
        max_degree=int(model["max_degree"]),
    )
    blocks = [base]
    if include_rbf_dynamics:
        rbf = predict_rbf_secondary(q_columns, mu, model).T
        blocks.append(rbf)
        if include_param_rbf_dynamics:
            mu = np.asarray(mu, dtype=np.float64).reshape(-1)
            for val in mu:
                blocks.append(float(val) * rbf)
    return np.hstack(blocks)


def rbf_continuous_feature_vector(q, mu, model):
    return rbf_continuous_feature_matrix(
        np.asarray(q, dtype=np.float64).reshape(-1, 1),
        mu,
        model,
        include_rbf_dynamics=bool(model["include_rbf_dynamics"]),
        include_param_rbf_dynamics=bool(model["include_param_rbf_dynamics"]),
    )[0]


def rhs_rbf_continuous(q, mu, model):
    if bool(model["include_rbf_dynamics"]):
        theta = rbf_continuous_feature_vector(q, mu, model)
    else:
        theta = continuous_feature_vector(
            q,
            mu,
            feature_mode=model["feature_mode"],
            include_param_linear=bool(model["include_param_linear"]),
            include_quadratic=bool(model["include_quadratic"]),
            include_higher=bool(model["include_higher"]),
            max_degree=int(model["max_degree"]),
        )
    theta_scaled = (theta - model["x_mean"]) / model["x_scale"]
    return model["operator"] @ theta_scaled


def rollout_rbf_continuous_rk4(q0, mu, dt, num_steps, model, max_norm=1e12):
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q_snaps = np.zeros((q0.size, int(num_steps) + 1), dtype=np.float64)
    q_snaps[:, 0] = q0
    q = q0.copy()
    stable_steps = int(num_steps)
    unstable_reason = ""
    h = float(dt)
    for istep in range(int(num_steps)):
        k1 = rhs_rbf_continuous(q, mu, model)
        k2 = rhs_rbf_continuous(q + 0.5 * h * k1, mu, model)
        k3 = rhs_rbf_continuous(q + 0.5 * h * k2, mu, model)
        k4 = rhs_rbf_continuous(q + h * k3, mu, model)
        q = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (not np.all(np.isfinite(q))) or np.linalg.norm(q) > float(max_norm):
            stable_steps = istep
            unstable_reason = f"unstable at step {istep + 1}"
            q_snaps[:, istep + 1 :] = np.nan
            break
        q_snaps[:, istep + 1] = q
    return q_snaps, stable_steps, unstable_reason


def save_rbf_manifold_model(model_path, **kwargs):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    arrays = {}
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
    arrays["model_family"] = np.asarray(RBF_MANIFOLD_MODEL_FAMILY)
    np.savez(model_path, **arrays)


def load_rbf_manifold_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"RBF manifold OpInf model not found at {model_path}. "
            "Run OpInf/stage1_fit_rbf_manifold_opinf.py first."
        )
    data = np.load(model_path, allow_pickle=False)
    model = {key: data[key] for key in data.files}
    family = str(np.asarray(model["model_family"]).item())
    if family != RBF_MANIFOLD_MODEL_FAMILY:
        raise ValueError(f"Unsupported model_family={family!r}; expected {RBF_MANIFOLD_MODEL_FAMILY!r}.")

    for key in ("feature_mode", "pod_basis_path", "rbf_kernel_name", "dynamics_feature_type"):
        if key in model:
            model[key] = str(np.asarray(model[key]).item())
    for key in ("num_primary", "num_secondary", "num_steps", "max_degree", "num_features", "num_rbf_centers"):
        if key in model:
            model[key] = int(np.asarray(model[key]).item())
    for key in (
        "dt",
        "dynamics_ridge",
        "rbf_epsilon",
        "rbf_ridge",
        "rbf_cv_relative_error",
        "relative_manifold_training_error",
        "relative_derivative_training_error",
        "energy_captured_primary",
        "energy_captured_total_basis",
        "rbf_feature_min",
        "rbf_feature_max",
        "duplicate_tol",
    ):
        if key in model:
            model[key] = float(np.asarray(model[key]).item())
    for key in (
        "include_param_linear",
        "include_quadratic",
        "include_higher",
        "rbf_include_mu",
        "include_rbf_dynamics",
        "include_param_rbf_dynamics",
    ):
        if key in model:
            model[key] = bool(int(np.asarray(model[key]).item()))
    for key in (
        "operator",
        "x_mean",
        "x_scale",
        "rbf_centers",
        "rbf_weights",
        "rbf_x_min",
        "rbf_x_span",
    ):
        if key in model:
            model[key] = np.asarray(model[key], dtype=np.float64)
    return model
