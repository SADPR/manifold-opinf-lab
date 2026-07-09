"""Utilities for GPR Nonlinear-Map MPOD-OpInf on the KdV benchmark."""

import os

import numpy as np
from scipy.optimize import minimize

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


MODEL_FAMILY = "kdv_gpr_nm_mpod_continuous_opinf"


def normalize_coordinates(q_samples, q_mean, q_scale):
    q_samples = np.asarray(q_samples, dtype=np.float64)
    q_mean = np.asarray(q_mean, dtype=np.float64).reshape(-1, 1)
    q_scale = np.asarray(q_scale, dtype=np.float64).reshape(-1, 1)
    if q_samples.ndim == 1:
        return ((q_samples.reshape(-1, 1) - q_mean) / q_scale)[:, 0]
    return (q_samples - q_mean) / q_scale


def squared_distances(q_a, q_b):
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    if q_a.ndim == 1:
        q_a = q_a[:, None]
    if q_b.ndim == 1:
        q_b = q_b[:, None]
    a_norm2 = np.sum(q_a**2, axis=0, keepdims=True).T
    b_norm2 = np.sum(q_b**2, axis=0, keepdims=True)
    return np.maximum(a_norm2 + b_norm2 - 2.0 * q_a.T @ q_b, 0.0)


def gp_kernel(q_a, q_b, kernel, epsilon, signal_variance=1.0):
    d2 = squared_distances(q_a, q_b)
    eps2_d2 = float(epsilon) ** 2 * d2
    kernel = str(kernel).lower()
    if kernel in {"gaussian", "se", "squared_exponential"}:
        values = np.exp(-eps2_d2)
    elif kernel in {"matern32", "matern3/2"}:
        radius = np.sqrt(eps2_d2)
        values = (1.0 + np.sqrt(3.0) * radius) * np.exp(-np.sqrt(3.0) * radius)
    elif kernel in {"matern52", "matern5/2"}:
        radius = np.sqrt(eps2_d2)
        values = (1.0 + np.sqrt(5.0) * radius + 5.0 * radius**2 / 3.0) * np.exp(-np.sqrt(5.0) * radius)
    else:
        raise ValueError(f"Unsupported GP kernel: {kernel}")
    return float(signal_variance) * values


def fit_gp_posterior(centers, secondary, kernel, epsilon, noise, jitter=1e-12, signal_variance=1.0):
    """Fit independent-output GP posterior means with shared kernel hyperparameters."""
    centers = np.asarray(centers, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    kernel_matrix = gp_kernel(centers, centers, kernel, epsilon, signal_variance=signal_variance)
    system = kernel_matrix + (float(noise) + float(jitter)) * np.eye(kernel_matrix.shape[0], dtype=np.float64)
    try:
        alpha = np.linalg.solve(system, secondary.T)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(system, secondary.T, rcond=None)[0]
    secondary_fit = (kernel_matrix @ alpha).T
    return alpha, secondary_fit, kernel_matrix


def gp_negative_log_marginal_likelihood(centers, secondary, kernel, epsilon, noise, jitter=1e-12, signal_variance=1.0):
    """Return the multi-output GP negative log marginal likelihood."""
    centers = np.asarray(centers, dtype=np.float64)
    y = np.asarray(secondary, dtype=np.float64).T
    n_samples, n_outputs = y.shape
    kernel_matrix = gp_kernel(centers, centers, kernel, epsilon, signal_variance=signal_variance)
    system = kernel_matrix + (float(noise) + float(jitter)) * np.eye(n_samples, dtype=np.float64)
    try:
        chol = np.linalg.cholesky(system)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    except np.linalg.LinAlgError:
        return np.inf
    data_fit = 0.5 * float(np.sum(y * alpha))
    logdet = float(np.sum(np.log(np.diag(chol))))
    constant = 0.5 * n_samples * n_outputs * np.log(2.0 * np.pi)
    return data_fit + n_outputs * logdet + constant


def _clip_to_bounds(value, bounds):
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower <= 0.0 or upper <= lower:
        raise ValueError(f"Invalid positive bounds: {bounds}")
    return min(max(float(value), lower), upper)


def optimize_gp_hyperparameters(
    centers,
    secondary,
    kernels=("matern32", "gaussian"),
    initial_epsilon=0.5,
    initial_noise=1e-6,
    initial_signal_variance=1.0,
    epsilon_bounds=(1e-3, 10.0),
    noise_bounds=(1e-12, 1e-2),
    signal_variance_bounds=(1e-6, 1e6),
    jitter=1e-12,
    maxiter=60,
):
    """Optimize shared GP hyperparameters by marginal likelihood."""
    if not kernels:
        raise ValueError("At least one GP kernel must be provided.")

    initial = np.log(
        [
            _clip_to_bounds(initial_epsilon, epsilon_bounds),
            _clip_to_bounds(initial_noise, noise_bounds),
            _clip_to_bounds(initial_signal_variance, signal_variance_bounds),
        ]
    )
    log_bounds = [
        (np.log(float(epsilon_bounds[0])), np.log(float(epsilon_bounds[1]))),
        (np.log(float(noise_bounds[0])), np.log(float(noise_bounds[1]))),
        (np.log(float(signal_variance_bounds[0])), np.log(float(signal_variance_bounds[1]))),
    ]

    best = None
    for kernel in kernels:
        kernel_name = str(kernel).lower()

        def objective(log_params):
            epsilon, noise, signal_variance = np.exp(log_params)
            return gp_negative_log_marginal_likelihood(
                centers,
                secondary,
                kernel_name,
                epsilon,
                noise,
                jitter=jitter,
                signal_variance=signal_variance,
            )

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-10},
        )
        epsilon, noise, signal_variance = np.exp(result.x)
        candidate = {
            "kernel": kernel_name,
            "epsilon": float(epsilon),
            "noise": float(noise),
            "signal_variance": float(signal_variance),
            "negative_log_marginal_likelihood": float(result.fun),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_nit": int(result.nit),
            "optimizer_nfev": int(result.nfev),
        }
        if best is None or candidate["negative_log_marginal_likelihood"] < best["negative_log_marginal_likelihood"]:
            best = candidate

    return best


def gp_predict_secondary(q_samples, centers, alpha, kernel, epsilon, q_mean, q_scale, signal_variance=1.0):
    squeeze = np.asarray(q_samples).ndim == 1
    normalized_q = normalize_coordinates(q_samples, q_mean, q_scale)
    k_star = gp_kernel(centers, normalized_q, kernel, epsilon, signal_variance=signal_variance)
    prediction = np.asarray(alpha, dtype=np.float64).T @ k_star
    return prediction[:, 0] if squeeze else prediction


def gp_posterior_variance(
    q_samples,
    centers,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    noise=0.0,
    jitter=1e-12,
    signal_variance=1.0,
    include_noise=False,
):
    """Return the latent GP posterior variance at reduced coordinates q_samples."""
    squeeze = np.asarray(q_samples).ndim == 1
    normalized_q = normalize_coordinates(q_samples, q_mean, q_scale)
    centers = np.asarray(centers, dtype=np.float64)
    k_star = gp_kernel(centers, normalized_q, kernel, epsilon, signal_variance=signal_variance)
    kernel_matrix = gp_kernel(centers, centers, kernel, epsilon, signal_variance=signal_variance)
    system = kernel_matrix + (float(noise) + float(jitter)) * np.eye(kernel_matrix.shape[0], dtype=np.float64)
    try:
        chol = np.linalg.cholesky(system)
        solved = np.linalg.solve(chol, k_star)
        variance = float(signal_variance) - np.sum(solved**2, axis=0)
    except np.linalg.LinAlgError:
        solved = np.linalg.solve(system, k_star)
        variance = float(signal_variance) - np.sum(k_star * solved, axis=0)
    if include_noise:
        variance = variance + float(noise)
    variance = np.maximum(variance, 0.0)
    return float(variance[0]) if squeeze else variance


def compute_gpr_nm_mpod_manifold(
    train_snapshots,
    num_modes,
    total_modes,
    kernel="matern32",
    epsilon=0.5,
    noise=1e-6,
    jitter=1e-12,
    signal_variance=1.0,
    optimize_hyperparameters=False,
    kernels=("matern32", "gaussian"),
    initial_epsilon=0.5,
    initial_noise=1e-6,
    initial_signal_variance=1.0,
    epsilon_bounds=(1e-3, 10.0),
    noise_bounds=(1e-12, 1e-2),
    signal_variance_bounds=(1e-6, 1e6),
    optimizer_maxiter=60,
):
    """Compute POD bases and learn secondary coordinates with GP regression."""
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
    secondary = basis_bar.T @ (centered - basis @ q)

    q_mean = np.mean(q, axis=1)
    q_scale = np.std(q, axis=1)
    q_scale = np.where(q_scale > 1e-12, q_scale, 1.0)
    centers = normalize_coordinates(q, q_mean, q_scale)

    if optimize_hyperparameters:
        optimizer_info = optimize_gp_hyperparameters(
            centers,
            secondary,
            kernels=tuple(kernels),
            initial_epsilon=initial_epsilon,
            initial_noise=initial_noise,
            initial_signal_variance=initial_signal_variance,
            epsilon_bounds=epsilon_bounds,
            noise_bounds=noise_bounds,
            signal_variance_bounds=signal_variance_bounds,
            jitter=jitter,
            maxiter=optimizer_maxiter,
        )
        kernel = optimizer_info["kernel"]
        epsilon = optimizer_info["epsilon"]
        noise = optimizer_info["noise"]
        signal_variance = optimizer_info["signal_variance"]
    else:
        optimizer_info = {
            "kernel": str(kernel).lower(),
            "epsilon": float(epsilon),
            "noise": float(noise),
            "signal_variance": float(signal_variance),
            "negative_log_marginal_likelihood": gp_negative_log_marginal_likelihood(
                centers,
                secondary,
                kernel,
                epsilon,
                noise,
                jitter=jitter,
                signal_variance=signal_variance,
            ),
            "optimizer_success": False,
            "optimizer_message": "not optimized",
            "optimizer_nit": 0,
            "optimizer_nfev": 0,
        }

    alpha, secondary_fit, kernel_matrix = fit_gp_posterior(
        centers,
        secondary,
        kernel,
        epsilon,
        noise,
        jitter=jitter,
        signal_variance=signal_variance,
    )
    reconstruction = u_ref[:, None] + basis @ q + basis_bar @ secondary_fit
    denom = max(np.linalg.norm(centered, "fro"), 1e-15)
    reconstruction_error = float(np.linalg.norm(train_snapshots - reconstruction, "fro") / denom)
    energy_captured = float(np.linalg.norm(basis @ q + basis_bar @ secondary_fit, "fro") ** 2 / denom**2)

    return {
        "basis": basis,
        "basis_bar": basis_bar,
        "sigma": sigma,
        "u_ref": u_ref,
        "q": q,
        "secondary": secondary,
        "secondary_fit": secondary_fit,
        "q_mean": q_mean,
        "q_scale": q_scale,
        "centers": centers,
        "alpha": alpha,
        "kernel": str(kernel).lower(),
        "epsilon": float(epsilon),
        "noise": float(noise),
        "jitter": float(jitter),
        "signal_variance": float(signal_variance),
        "negative_log_marginal_likelihood": float(optimizer_info["negative_log_marginal_likelihood"]),
        "optimizer_success": bool(optimizer_info["optimizer_success"]),
        "optimizer_message": str(optimizer_info["optimizer_message"]),
        "optimizer_nit": int(optimizer_info["optimizer_nit"]),
        "optimizer_nfev": int(optimizer_info["optimizer_nfev"]),
        "reconstruction": reconstruction,
        "relative_reconstruction_error": reconstruction_error,
        "energy_captured": energy_captured,
    }


def reconstruct_gpr_nm_mpod_snapshots(q, basis, basis_bar, alpha, u_ref, centers, kernel, epsilon, q_mean, q_scale, signal_variance=1.0):
    q = np.asarray(q, dtype=np.float64)
    secondary = gp_predict_secondary(q, centers, alpha, kernel, epsilon, q_mean, q_scale, signal_variance=signal_variance)
    return (
        np.asarray(u_ref, dtype=np.float64)[:, None]
        + np.asarray(basis, dtype=np.float64) @ q
        + np.asarray(basis_bar, dtype=np.float64) @ secondary
    )


def gpr_nm_mpod_feature_vector(
    q,
    centers,
    alpha,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    signal_variance=1.0,
    include_quadratic=True,
    include_full_quadratic=False,
):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    secondary = gp_predict_secondary(q, centers, alpha, kernel, epsilon, q_mean, q_scale, signal_variance=signal_variance)
    if bool(include_full_quadratic):
        return np.concatenate(
            (
                [1.0],
                q,
                compact_quadratic_features(q),
                secondary,
                np.kron(q, secondary),
                compact_quadratic_features(secondary),
            )
        )
    if bool(include_quadratic):
        return np.concatenate(([1.0], q, compact_quadratic_features(q), secondary))
    return np.concatenate(([1.0], q, secondary))


def gpr_nm_mpod_feature_matrix(
    q_samples,
    centers,
    alpha,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    signal_variance=1.0,
    include_quadratic=True,
    include_full_quadratic=False,
):
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim != 2:
        raise ValueError("q_samples must have shape (r, m).")
    return np.vstack(
        [
            gpr_nm_mpod_feature_vector(
                q_samples[:, j],
                centers,
                alpha,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                signal_variance,
                include_quadratic=include_quadratic,
                include_full_quadratic=include_full_quadratic,
            )
            for j in range(q_samples.shape[1])
        ]
    )


def fit_gpr_nm_mpod_continuous_operator(
    q,
    qdot,
    centers,
    alpha,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    signal_variance=1.0,
    ridge_c=0.0,
    ridge_a=0.0,
    ridge_h=0.0,
    ridge_gpr=0.0,
    include_quadratic=True,
    include_full_quadratic=False,
):
    """Fit GPR-NM-MPOD OpInf dynamics.

    Feature order:
      lifted linear:  [1, q, z]
      latent closure: [1, q, q_quad, z]
      full quadratic: [1, q, q_quad, z, q kron z, z_quad]
    """
    q = np.asarray(q, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    if q.ndim != 2 or qdot.ndim != 2:
        raise ValueError("q and qdot must be 2D arrays.")
    if q.shape != qdot.shape:
        raise ValueError(f"q/qdot shape mismatch: {q.shape} vs {qdot.shape}.")

    include_full_quadratic = bool(include_full_quadratic)
    include_quadratic = bool(include_quadratic) or include_full_quadratic
    theta = gpr_nm_mpod_feature_matrix(
        q,
        centers,
        alpha,
        kernel,
        epsilon,
        q_mean,
        q_scale,
        signal_variance,
        include_quadratic=include_quadratic,
        include_full_quadratic=include_full_quadratic,
    )
    target = qdot.T
    r = q.shape[0]
    n_const = 1
    n_linear = r
    n_quadratic = r * (r + 1) // 2 if include_quadratic else 0
    n_gpr = alpha.shape[1]
    n_mixed = r * n_gpr if include_full_quadratic else 0
    n_secondary_quadratic = n_gpr * (n_gpr + 1) // 2 if include_full_quadratic else 0
    penalties = np.concatenate(
        [
            np.full(n_const, float(ridge_c), dtype=np.float64),
            np.full(n_linear, float(ridge_a), dtype=np.float64),
            np.full(n_quadratic, float(ridge_h), dtype=np.float64),
            np.full(n_gpr, float(ridge_gpr), dtype=np.float64),
            np.full(n_mixed, float(ridge_h), dtype=np.float64),
            np.full(n_secondary_quadratic, float(ridge_h), dtype=np.float64),
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
        "ridge_gpr": float(ridge_gpr),
        "include_quadratic": include_quadratic,
        "include_full_quadratic": include_full_quadratic,
        "num_quadratic_features": int(n_quadratic),
        "num_mixed_features": int(n_mixed),
        "num_secondary_quadratic_features": int(n_secondary_quadratic),
    }


def rhs(
    q,
    coeffs,
    centers,
    alpha,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    signal_variance=1.0,
    include_quadratic=True,
    include_full_quadratic=False,
):
    return np.asarray(coeffs, dtype=np.float64) @ gpr_nm_mpod_feature_vector(
        q,
        centers,
        alpha,
        kernel,
        epsilon,
        q_mean,
        q_scale,
        signal_variance,
        include_quadratic=include_quadratic,
        include_full_quadratic=include_full_quadratic,
    )


def rollout_rk4(
    q0,
    times,
    coeffs,
    centers,
    alpha,
    kernel,
    epsilon,
    q_mean,
    q_scale,
    signal_variance=1.0,
    max_norm=np.inf,
    substeps=1,
    include_quadratic=True,
    include_full_quadratic=False,
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
            k1 = rhs(
                y_next,
                coeffs,
                centers,
                alpha,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                signal_variance,
                include_quadratic=include_quadratic,
                include_full_quadratic=include_full_quadratic,
            )
            k2 = rhs(
                y_next + 0.5 * h * k1,
                coeffs,
                centers,
                alpha,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                signal_variance,
                include_quadratic=include_quadratic,
                include_full_quadratic=include_full_quadratic,
            )
            k3 = rhs(
                y_next + 0.5 * h * k2,
                coeffs,
                centers,
                alpha,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                signal_variance,
                include_quadratic=include_quadratic,
                include_full_quadratic=include_full_quadratic,
            )
            k4 = rhs(
                y_next + h * k3,
                coeffs,
                centers,
                alpha,
                kernel,
                epsilon,
                q_mean,
                q_scale,
                signal_variance,
                include_quadratic=include_quadratic,
                include_full_quadratic=include_full_quadratic,
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
        raise FileNotFoundError(f"GPR-NM-MPOD-OpInf model not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    for key in ("model_family", "snapshot_file", "regularizer_convention", "kernel", "model_variant", "operator_mode"):
        if key in data:
            data[key] = str(np.asarray(data[key]).item())
    for key in (
        "num_modes",
        "total_modes",
        "num_secondary",
        "num_features",
        "num_quadratic_features",
        "num_gpr_features",
        "num_mixed_features",
        "num_secondary_quadratic_features",
        "unstable_index",
        "rk4_substeps",
    ):
        if key in data:
            data[key] = int(np.asarray(data[key]).item())
    if "include_quadratic" in data:
        data["include_quadratic"] = bool(np.asarray(data["include_quadratic"]).item())
    else:
        data["include_quadratic"] = True
    if "include_full_quadratic" in data:
        data["include_full_quadratic"] = bool(np.asarray(data["include_full_quadratic"]).item())
    else:
        data["include_full_quadratic"] = False
    for key in (
        "dt",
        "train_final_time",
        "epsilon",
        "noise",
        "jitter",
        "signal_variance",
        "ridge_c",
        "ridge_a",
        "ridge_h",
        "ridge_gpr",
        "energy_captured",
        "relative_reconstruction_error",
        "relative_derivative_error",
        "training_rollout_error",
    ):
        if key in data:
            data[key] = float(np.asarray(data[key]).item())
    return data
