"""Utilities for RBF Nonlinear-Map MPOD-OpInf on the KdV benchmark."""

import os

import numpy as np

from standard_opinf_utils import (
    compact_quadratic_features,
    fourth_order_time_derivative,
    load_fom_dataset,
    plot_error_history,
    plot_snapshot_comparison,
    plot_spacetime_comparison,
    project_snapshots,
    relative_error_history,
    relative_state_error,
)


MODEL_FAMILY = "kdv_rbf_nm_mpod_continuous_opinf"


def normalize_coordinates(q_samples, q_mean, q_scale):
    q_samples = np.asarray(q_samples, dtype=np.float64)
    q_mean = np.asarray(q_mean, dtype=np.float64).reshape(-1, 1)
    q_scale = np.asarray(q_scale, dtype=np.float64).reshape(-1, 1)
    if q_samples.ndim == 1:
        return ((q_samples.reshape(-1, 1) - q_mean) / q_scale)[:, 0]
    return (q_samples - q_mean) / q_scale


def squared_distances(normalized_q, centers):
    normalized_q = np.asarray(normalized_q, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if normalized_q.ndim == 1:
        normalized_q = normalized_q[:, None]
        squeeze = True
    else:
        squeeze = False
    q_norm2 = np.sum(normalized_q**2, axis=0, keepdims=True)
    c_norm2 = np.sum(centers**2, axis=0, keepdims=True).T
    d2 = np.maximum(c_norm2 + q_norm2 - 2.0 * centers.T @ normalized_q, 0.0)
    return d2[:, 0] if squeeze else d2


def rbf_features_from_normalized(normalized_q, centers, kernel, epsilon):
    d2 = squared_distances(normalized_q, centers)
    eps2_d2 = float(epsilon) ** 2 * d2
    kernel = str(kernel).lower()
    if kernel in {"gaussian", "gauss"}:
        phi = np.exp(-eps2_d2)
    elif kernel in {"imq", "inverse_multiquadric"}:
        phi = 1.0 / np.sqrt(1.0 + eps2_d2)
    elif kernel in {"iq", "inverse_quadratic"}:
        phi = 1.0 / (1.0 + eps2_d2)
    elif kernel in {"matern32", "matern3/2"}:
        radius = np.sqrt(eps2_d2)
        phi = (1.0 + np.sqrt(3.0) * radius) * np.exp(-np.sqrt(3.0) * radius)
    else:
        raise ValueError(f"Unsupported RBF kernel: {kernel}")
    return np.asarray(phi, dtype=np.float64)


def rbf_features(q_samples, centers, kernel, epsilon, q_mean, q_scale):
    normalized = normalize_coordinates(q_samples, q_mean, q_scale)
    return rbf_features_from_normalized(normalized, centers, kernel, epsilon)


def select_center_indices(num_samples, center_stride=1, max_centers=0):
    stride = max(1, int(center_stride))
    indices = np.arange(0, int(num_samples), stride, dtype=np.int64)
    max_centers = int(max_centers)
    if max_centers > 0 and indices.size > max_centers:
        pick = np.linspace(0, indices.size - 1, max_centers)
        indices = indices[np.unique(np.round(pick).astype(np.int64))]
    return indices


def compute_rbf_nm_mpod_manifold(
    train_snapshots,
    num_modes,
    total_modes,
    kernel,
    epsilon,
    rbf_ridge,
    center_stride=1,
    max_centers=0,
):
    """Compute POD primary/secondary bases and fit secondary RBF coordinates."""
    train_snapshots = np.asarray(train_snapshots, dtype=np.float64)
    r = int(num_modes)
    total = int(total_modes)
    if total <= r:
        raise ValueError("total_modes must be larger than num_modes.")
    if total > min(train_snapshots.shape):
        raise ValueError(f"total_modes must be <= {min(train_snapshots.shape)}, got {total}.")

    u_ref = np.mean(train_snapshots, axis=1)
    centered = train_snapshots - u_ref[:, None]
    left, sigma, _ = np.linalg.svd(centered, full_matrices=False)
    basis = left[:, :r]
    basis_bar = left[:, r:total]
    q = basis.T @ centered
    residual_secondary = basis_bar.T @ (centered - basis @ q)

    q_mean = np.mean(q, axis=1)
    q_scale = np.std(q, axis=1)
    q_scale = np.where(q_scale > 1e-12, q_scale, 1.0)
    normalized_q = normalize_coordinates(q, q_mean, q_scale)
    center_indices = select_center_indices(q.shape[1], center_stride=center_stride, max_centers=max_centers)
    centers = normalized_q[:, center_indices]
    phi = rbf_features_from_normalized(normalized_q, centers, kernel, epsilon)

    normal = phi @ phi.T + float(rbf_ridge) * np.eye(phi.shape[0], dtype=np.float64)
    rhs = residual_secondary @ phi.T
    try:
        weights = np.linalg.solve(normal, rhs.T).T
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(normal, rhs.T, rcond=None)[0].T

    reconstruction = u_ref[:, None] + basis @ q + basis_bar @ (weights @ phi)
    denom = max(np.linalg.norm(centered, "fro"), 1e-15)
    reconstruction_error = float(np.linalg.norm(train_snapshots - reconstruction, "fro") / denom)
    energy_captured = float(np.linalg.norm(basis @ q + basis_bar @ (weights @ phi), "fro") ** 2 / denom**2)

    return {
        "basis": basis,
        "basis_bar": basis_bar,
        "sigma": sigma,
        "u_ref": u_ref,
        "q": q,
        "q_mean": q_mean,
        "q_scale": q_scale,
        "center_indices": center_indices,
        "centers": centers,
        "phi": phi,
        "weights": weights,
        "kernel": str(kernel).lower(),
        "epsilon": float(epsilon),
        "rbf_ridge": float(rbf_ridge),
        "reconstruction": reconstruction,
        "relative_reconstruction_error": reconstruction_error,
        "energy_captured": energy_captured,
    }


def reconstruct_rbf_nm_mpod_snapshots(q, basis, basis_bar, weights, u_ref, centers, kernel, epsilon, q_mean, q_scale):
    q = np.asarray(q, dtype=np.float64)
    phi = rbf_features(q, centers, kernel, epsilon, q_mean, q_scale)
    return (
        np.asarray(u_ref, dtype=np.float64)[:, None]
        + np.asarray(basis, dtype=np.float64) @ q
        + np.asarray(basis_bar, dtype=np.float64) @ (np.asarray(weights, dtype=np.float64) @ phi)
    )


def rbf_nm_mpod_feature_vector(q, centers, kernel, epsilon, q_mean, q_scale, include_quadratic=True):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    phi = rbf_features(q, centers, kernel, epsilon, q_mean, q_scale)
    if bool(include_quadratic):
        return np.concatenate(([1.0], q, compact_quadratic_features(q), phi))
    return np.concatenate(([1.0], q, phi))


def rbf_nm_mpod_feature_matrix(q_samples, centers, kernel, epsilon, q_mean, q_scale, include_quadratic=True):
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim != 2:
        raise ValueError("q_samples must have shape (r, m).")
    return np.vstack(
        [
            rbf_nm_mpod_feature_vector(
                q_samples[:, j],
                centers,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                include_quadratic=include_quadratic,
            )
            for j in range(q_samples.shape[1])
        ]
    )


def split_rbf_nm_mpod_feature_counts(num_modes, num_rbf_features, include_quadratic=True):
    r = int(num_modes)
    n_quadratic = r * (r + 1) // 2 if bool(include_quadratic) else 0
    return 1, r, n_quadratic, int(num_rbf_features)


def fit_rbf_nm_mpod_continuous_operator(
    q,
    qdot,
    centers,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    ridge_c=0.0,
    ridge_a=0.0,
    ridge_h=0.0,
    ridge_rbf=0.0,
    include_quadratic=True,
):
    """Fit dq/dt = c + A q + H q_quad + P phi(q), or omit H if requested."""
    q = np.asarray(q, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    if q.ndim != 2 or qdot.ndim != 2:
        raise ValueError("q and qdot must be 2D arrays.")
    if q.shape != qdot.shape:
        raise ValueError(f"q/qdot shape mismatch: {q.shape} vs {qdot.shape}.")

    include_quadratic = bool(include_quadratic)
    theta = rbf_nm_mpod_feature_matrix(
        q, centers, kernel, epsilon, q_mean, q_scale, include_quadratic=include_quadratic
    )
    target = qdot.T
    n_const, n_linear, n_quadratic, n_rbf = split_rbf_nm_mpod_feature_counts(
        q.shape[0], centers.shape[1], include_quadratic=include_quadratic
    )
    penalties = np.concatenate(
        [
            np.full(n_const, float(ridge_c), dtype=np.float64),
            np.full(n_linear, float(ridge_a), dtype=np.float64),
            np.full(n_quadratic, float(ridge_h), dtype=np.float64),
            np.full(n_rbf, float(ridge_rbf), dtype=np.float64),
        ]
    )
    if np.any(penalties < 0.0):
        raise ValueError("Ridge regularization parameters must be nonnegative.")

    if np.any(penalties > 0.0):
        theta_aug = np.vstack([theta, np.diag(penalties)])
        target_aug = np.vstack([target, np.zeros((penalties.size, target.shape[1]), dtype=np.float64)])
    else:
        theta_aug = theta
        target_aug = target

    solution, residuals, rank, singular_values = np.linalg.lstsq(theta_aug, target_aug, rcond=None)
    coeffs = solution.T
    fit = coeffs @ theta.T
    rel_derivative_error = np.linalg.norm(fit - qdot) / max(np.linalg.norm(qdot), 1e-15)
    return {
        "coeffs": coeffs,
        "theta_shape": theta.shape,
        "rank": int(rank),
        "singular_values": singular_values,
        "relative_derivative_error": float(rel_derivative_error),
        "ridge_c": float(ridge_c),
        "ridge_a": float(ridge_a),
        "ridge_h": float(ridge_h),
        "ridge_rbf": float(ridge_rbf),
        "include_quadratic": include_quadratic,
        "num_quadratic_features": int(n_quadratic),
    }


def rhs(q, coeffs, centers, kernel, epsilon, q_mean, q_scale, include_quadratic=True):
    return np.asarray(coeffs, dtype=np.float64) @ rbf_nm_mpod_feature_vector(
        q, centers, kernel, epsilon, q_mean, q_scale, include_quadratic=include_quadratic
    )


def rollout_rk4(
    q0,
    times,
    coeffs,
    centers,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    max_norm=np.inf,
    substeps=1,
    include_quadratic=True,
):
    times = np.asarray(times, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    substeps = max(1, int(substeps))
    q = np.empty((q0.size, times.size), dtype=np.float64)
    q[:, 0] = q0
    unstable = False
    unstable_index = -1
    for j in range(times.size - 1):
        dt = float(times[j + 1] - times[j])
        h = dt / float(substeps)
        y_next = q[:, j].copy()
        for _ in range(substeps):
            k1 = rhs(y_next, coeffs, centers, kernel, epsilon, q_mean, q_scale, include_quadratic=include_quadratic)
            k2 = rhs(
                y_next + 0.5 * h * k1,
                coeffs,
                centers,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                include_quadratic=include_quadratic,
            )
            k3 = rhs(
                y_next + 0.5 * h * k2,
                coeffs,
                centers,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                include_quadratic=include_quadratic,
            )
            k4 = rhs(
                y_next + h * k3,
                coeffs,
                centers,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                include_quadratic=include_quadratic,
            )
            y_next = y_next + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(y_next)) or np.linalg.norm(y_next) > float(max_norm):
            unstable = True
            unstable_index = j + 1
            q[:, j + 1:] = np.nan
            break
        q[:, j + 1] = y_next
    return q, unstable, unstable_index


def save_model(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, model_family=np.asarray(MODEL_FAMILY), **arrays)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"RBF-NM-MPOD-OpInf model not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    for key in ("model_family", "snapshot_file", "regularizer_convention", "kernel", "model_variant"):
        if key in data:
            data[key] = str(np.asarray(data[key]).item())
    for key in (
        "num_modes",
        "total_modes",
        "num_secondary",
        "num_features",
        "num_quadratic_features",
        "num_rbf_features",
        "unstable_index",
        "rk4_substeps",
    ):
        if key in data:
            data[key] = int(np.asarray(data[key]).item())
    if "include_quadratic" in data:
        data["include_quadratic"] = bool(np.asarray(data["include_quadratic"]).item())
    else:
        data["include_quadratic"] = True
    for key in (
        "dt",
        "train_final_time",
        "epsilon",
        "rbf_ridge",
        "ridge_c",
        "ridge_a",
        "ridge_h",
        "ridge_rbf",
        "energy_captured",
        "relative_reconstruction_error",
        "relative_derivative_error",
        "training_rollout_error",
    ):
        if key in data:
            data[key] = float(np.asarray(data[key]).item())
    return data
