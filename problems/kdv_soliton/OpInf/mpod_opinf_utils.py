"""Utilities for POD-based polynomial-manifold OpInf on the KdV benchmark."""

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


MODEL_FAMILY = "kdv_mpod_polynomial_continuous_opinf"


def polynomial_embedding(q_samples, degree):
    """Return g(q) = [q^2; ...; q^p] using elementwise powers."""
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim == 1:
        q_samples = q_samples[:, None]
        squeeze = True
    elif q_samples.ndim == 2:
        squeeze = False
    else:
        raise ValueError("q_samples must have shape (r,) or (r, m).")
    degree = int(degree)
    if degree < 2:
        raise ValueError("degree must be >= 2.")
    blocks = [q_samples**power for power in range(2, degree + 1)]
    embedded = np.vstack(blocks)
    return embedded[:, 0] if squeeze else embedded


def higher_monomial_exponents(num_modes, degree):
    """Return exponent rows for ghat(q), the degree >= 3 MPOD OpInf features."""
    r = int(num_modes)
    p = int(degree)
    if r < 1:
        raise ValueError("num_modes must be positive.")
    if p < 2:
        raise ValueError("degree must be >= 2.")

    exponents = set()

    def unit(index, power=1):
        row = [0] * r
        row[int(index)] = int(power)
        return row

    # Linear full-order terms acting on the manifold polynomial g(q).
    for power in range(3, p + 1):
        for i in range(r):
            exponents.add(tuple(unit(i, power)))

    # Quadratic FOM interactions between V q and Vbar Xi g(q).
    for i in range(r):
        for j in range(r):
            for power in range(2, p + 1):
                row = unit(i, 1)
                row[j] += power
                if sum(row) >= 3:
                    exponents.add(tuple(row))

    # Quadratic FOM interactions inside Vbar Xi g(q).
    g_terms = []
    for power in range(2, p + 1):
        for i in range(r):
            g_terms.append(tuple(unit(i, power)))
    for a, first in enumerate(g_terms):
        for second in g_terms[a:]:
            row = tuple(first[i] + second[i] for i in range(r))
            if sum(row) >= 3:
                exponents.add(row)

    return np.asarray(sorted(exponents, key=lambda item: (sum(item), item)), dtype=np.int64)


def evaluate_monomials(q, exponents):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    exponents = np.asarray(exponents, dtype=np.int64)
    values = np.ones(exponents.shape[0], dtype=np.float64)
    for j in range(q.size):
        powers = exponents[:, j]
        active = powers != 0
        if np.any(active):
            values[active] *= q[j] ** powers[active]
    return values


def mpod_feature_vector(q, exponents):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return np.concatenate(([1.0], q, compact_quadratic_features(q), evaluate_monomials(q, exponents)))


def mpod_feature_matrix(q_samples, exponents):
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim != 2:
        raise ValueError("q_samples must have shape (r, m).")
    return np.vstack([mpod_feature_vector(q_samples[:, j], exponents) for j in range(q_samples.shape[1])])


def split_mpod_feature_counts(num_modes, num_higher_features):
    r = int(num_modes)
    return 1, r, r * (r + 1) // 2, int(num_higher_features)


def compute_mpod_manifold(train_snapshots, num_modes, total_modes, degree, gamma):
    """Compute Algorithm-1 POD-based polynomial manifold data."""
    train_snapshots = np.asarray(train_snapshots, dtype=np.float64)
    r = int(num_modes)
    total = int(total_modes)
    p = int(degree)
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
    g = polynomial_embedding(q, p)
    projection_error = centered - basis @ q

    normal = g @ g.T + float(gamma) * np.eye(g.shape[0], dtype=np.float64)
    rhs = basis_bar.T @ projection_error @ g.T
    xi = np.linalg.solve(normal, rhs.T).T
    reconstruction = u_ref[:, None] + basis @ q + basis_bar @ (xi @ g)

    denom = max(np.linalg.norm(centered), 1e-15)
    reconstruction_error = float(np.linalg.norm(train_snapshots - reconstruction) / denom)
    energy_captured = float(np.linalg.norm(basis @ q + basis_bar @ (xi @ g), "fro") ** 2 / denom**2)

    return {
        "basis": basis,
        "basis_bar": basis_bar,
        "sigma": sigma,
        "u_ref": u_ref,
        "q": q,
        "poly": g,
        "xi": xi,
        "reconstruction": reconstruction,
        "relative_reconstruction_error": reconstruction_error,
        "energy_captured": energy_captured,
    }


def reconstruct_mpod_snapshots(q, basis, basis_bar, xi, u_ref, degree):
    q = np.asarray(q, dtype=np.float64)
    g = polynomial_embedding(q, int(degree))
    return (
        np.asarray(u_ref, dtype=np.float64)[:, None]
        + np.asarray(basis, dtype=np.float64) @ q
        + np.asarray(basis_bar, dtype=np.float64) @ (np.asarray(xi, dtype=np.float64) @ g)
    )


def fit_mpod_continuous_operator(
    q,
    qdot,
    exponents,
    ridge_c=0.0,
    ridge_a=0.0,
    ridge_h=0.0,
    ridge_p=None,
):
    """Fit dq/dt = c + A q + H q_quad + P ghat(q).

    The ridge values are direct Tikhonov weights, matching the convention used
    by opinf.lstsq.L2Solver/TikhonovSolver. The solved objective contains
    ridge_*^2 times the corresponding operator norm.
    """
    q = np.asarray(q, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    if q.ndim != 2 or qdot.ndim != 2:
        raise ValueError("q and qdot must be 2D arrays.")
    if q.shape != qdot.shape:
        raise ValueError(f"q/qdot shape mismatch: {q.shape} vs {qdot.shape}.")

    if ridge_p is None:
        ridge_p = ridge_h

    theta = mpod_feature_matrix(q, exponents)
    target = qdot.T
    n_const, n_linear, n_quadratic, n_higher = split_mpod_feature_counts(q.shape[0], len(exponents))
    regularizer_weights = np.concatenate(
        [
            np.full(n_const, float(ridge_c), dtype=np.float64),
            np.full(n_linear, float(ridge_a), dtype=np.float64),
            np.full(n_quadratic, float(ridge_h), dtype=np.float64),
            np.full(n_higher, float(ridge_p), dtype=np.float64),
        ]
    )
    if np.any(regularizer_weights < 0.0):
        raise ValueError("Ridge regularization parameters must be nonnegative.")

    if np.any(regularizer_weights > 0.0):
        theta_aug = np.vstack([theta, np.diag(regularizer_weights)])
        target_aug = np.vstack([target, np.zeros((regularizer_weights.size, target.shape[1]), dtype=np.float64)])
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
        "ridge_p": float(ridge_p),
    }


def rhs(q, coeffs, exponents):
    return np.asarray(coeffs, dtype=np.float64) @ mpod_feature_vector(q, exponents)


def rollout_rk4(q0, times, coeffs, exponents, max_norm=np.inf, substeps=1):
    times = np.asarray(times, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    substeps = max(1, int(substeps))
    q = np.empty((q0.size, times.size), dtype=np.float64)
    q[:, 0] = q0
    unstable = False
    unstable_index = -1
    for j in range(times.size - 1):
        dt = float(times[j + 1] - times[j])
        y = q[:, j]
        h = dt / float(substeps)
        y_next = y.copy()
        for _ in range(substeps):
            k1 = rhs(y_next, coeffs, exponents)
            k2 = rhs(y_next + 0.5 * h * k1, coeffs, exponents)
            k3 = rhs(y_next + 0.5 * h * k2, coeffs, exponents)
            k4 = rhs(y_next + h * k3, coeffs, exponents)
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
        raise FileNotFoundError(f"MPOD-OpInf model not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    for key in ("model_family", "snapshot_file", "regularizer_convention"):
        if key in data:
            data[key] = str(np.asarray(data[key]).item())
    for key in ("num_modes", "total_modes", "num_secondary", "degree", "num_features", "num_higher_features", "unstable_index"):
        if key in data:
            data[key] = int(np.asarray(data[key]).item())
    for key in (
        "dt",
        "train_final_time",
        "gamma",
        "ridge_c",
        "ridge_a",
        "ridge_h",
        "ridge_p",
        "energy_captured",
        "relative_reconstruction_error",
        "relative_derivative_error",
        "training_rollout_error",
    ):
        if key in data:
            data[key] = float(np.asarray(data[key]).item())
    return data
