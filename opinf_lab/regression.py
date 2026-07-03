"""Regression routines shared by continuous and discrete OpInf scripts."""

import numpy as np


def fit_operator(theta, targets, ridge=1e-8, penalize_intercept=False, error_name="relative_training_error"):
    """Fit ``targets ~= theta @ operator.T`` with column scaling and ridge.

    Returns the operator in the conventional OpInf shape ``(n_outputs,
    n_features)`` plus the feature scaling used at evaluation time.
    """
    theta = np.asarray(theta, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if theta.ndim != 2 or targets.ndim != 2:
        raise ValueError("theta and targets must be 2D arrays.")
    if theta.shape[0] != targets.shape[0]:
        raise ValueError(f"Sample mismatch: theta {theta.shape}, targets {targets.shape}.")

    x_mean = np.zeros(theta.shape[1], dtype=np.float64)
    x_scale = np.ones(theta.shape[1], dtype=np.float64)
    if theta.shape[1] > 1:
        x_mean[1:] = np.mean(theta[:, 1:], axis=0)
        x_scale[1:] = np.std(theta[:, 1:], axis=0)
        x_scale[1:] = np.where(x_scale[1:] > 1e-14, x_scale[1:], 1.0)
    theta_scaled = (theta - x_mean[None, :]) / x_scale[None, :]

    reg = float(ridge)
    if reg < 0.0:
        raise ValueError(f"ridge must be nonnegative, got {ridge}.")

    n_samples, n_features = theta_scaled.shape
    use_dual = bool(reg > 0.0 and n_features > n_samples)
    if use_dual and not penalize_intercept and n_features > 1:
        # The first column is the constant feature and all other columns are
        # centered by construction. Therefore the unpenalized intercept
        # decouples exactly from the ridge-penalized coefficients.
        z = theta_scaled[:, 1:]
        target_mean = np.mean(targets, axis=0)
        centered_targets = targets - target_mean[None, :]
        lhs = z @ z.T + reg * np.eye(n_samples, dtype=np.float64)
        try:
            alpha = np.linalg.solve(lhs, centered_targets)
            solve_method = "dual_solve_unpenalized_intercept"
        except np.linalg.LinAlgError:
            alpha, *_ = np.linalg.lstsq(lhs, centered_targets, rcond=None)
            solve_method = "dual_lstsq_unpenalized_intercept"
        operator_t = np.vstack((target_mean[None, :], z.T @ alpha))
    elif use_dual:
        lhs = theta_scaled @ theta_scaled.T + reg * np.eye(n_samples, dtype=np.float64)
        try:
            alpha = np.linalg.solve(lhs, targets)
            solve_method = "dual_solve"
        except np.linalg.LinAlgError:
            alpha, *_ = np.linalg.lstsq(lhs, targets, rcond=None)
            solve_method = "dual_lstsq"
        operator_t = theta_scaled.T @ alpha
    else:
        lhs = theta_scaled.T @ theta_scaled
        if reg > 0.0:
            reg_diag = np.ones(lhs.shape[0], dtype=np.float64)
            if not penalize_intercept and reg_diag.size:
                reg_diag[0] = 0.0
            lhs = lhs + reg * np.diag(reg_diag)
        rhs = theta_scaled.T @ targets

        try:
            operator_t = np.linalg.solve(lhs, rhs)
            solve_method = "solve"
        except np.linalg.LinAlgError:
            operator_t, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            solve_method = "lstsq"

    operator = operator_t.T
    pred = theta_scaled @ operator.T
    denom = np.linalg.norm(targets)
    rel_error = np.linalg.norm(targets - pred) / (denom if denom > 0.0 else 1.0)
    return {
        "operator": operator,
        "x_mean": x_mean,
        "x_scale": x_scale,
        error_name: float(rel_error),
        "solve_method": solve_method,
    }
