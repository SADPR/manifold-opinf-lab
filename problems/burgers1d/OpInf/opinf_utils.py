"""Shared utilities for discrete-time operator inference ROMs."""

import os
import sys

import numpy as np


LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)

from opinf_lab.features import (
    compact_monomial_matrix as _compact_monomial_matrix,
    compact_monomials as _compact_monomials,
    polynomial_parameter_features,
)
from opinf_lab.pod import project_snapshots as _project_snapshots
from opinf_lab.pod import reconstruct_snapshots as _reconstruct_snapshots
from opinf_lab.regression import fit_operator


FEATURE_MODE = "poly2_mu"
MODEL_FAMILY = "linear_discrete_parametric"
QUADRATIC_MODEL_FAMILY = "quadratic_discrete_parametric"


def parameter_features(mu, feature_mode=FEATURE_MODE):
    """Return parameter features used to make the linear operator parametric."""
    if feature_mode != FEATURE_MODE:
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}.")
    return polynomial_parameter_features(mu, degree=2, include_constant=False)


def design_vector(q, mu, feature_mode=FEATURE_MODE):
    """
    Build theta(q, mu) for q_{k+1} = W theta(q_k, mu).

    The state dependence is linear only. Parameter-state products make the
    learned linear map parameter-dependent, but no q_i q_j terms are included.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    eta = parameter_features(mu, feature_mode=feature_mode)
    return np.concatenate(([1.0], eta, q, np.kron(eta, q)))


def design_matrix(q_columns, mu, feature_mode=FEATURE_MODE):
    """Vectorized design matrix for reduced states stored as columns."""
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim != 2:
        raise ValueError(f"q_columns must be 2D, got shape {q_columns.shape}.")
    n_red, n_samples = q_columns.shape
    eta = parameter_features(mu, feature_mode=feature_mode)
    n_eta = eta.size
    theta = np.empty((n_samples, 1 + n_eta + n_red + n_eta * n_red), dtype=np.float64)
    theta[:, 0] = 1.0
    theta[:, 1:1 + n_eta] = eta[None, :]
    q_t = q_columns.T
    q0 = 1 + n_eta
    q1 = q0 + n_red
    theta[:, q0:q1] = q_t
    col = q1
    for val in eta:
        theta[:, col:col + n_red] = val * q_t
        col += n_red
    return theta


def compact_quadratic_terms(q):
    """Return upper-triangular products q_i q_j for i <= j."""
    return _compact_monomials(q, degree=2)


def compact_quadratic_matrix(q_columns):
    """Vectorized upper-triangular products for reduced states stored as columns."""
    return _compact_monomial_matrix(q_columns, degree=2)


def design_vector_quadratic(
    q,
    mu,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_param_quad=True,
):
    """
    Build theta(q, mu) for q_{k+1} = W theta(q_k, mu).

    This extends the linear design with compact quadratic reduced-state terms
    and parameter-quadratic products.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    eta = parameter_features(mu, feature_mode=feature_mode)
    quad = compact_quadratic_terms(q)
    parts = [[1.0], eta, q]
    if include_param_linear:
        parts.append(np.kron(eta, q))
    parts.append(quad)
    if include_param_quad:
        parts.append(np.kron(eta, quad))
    return np.concatenate(parts)


def design_matrix_quadratic(
    q_columns,
    mu,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_param_quad=True,
):
    """Vectorized quadratic design matrix for reduced states stored as columns."""
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim != 2:
        raise ValueError(f"q_columns must be 2D, got shape {q_columns.shape}.")
    n_red, n_samples = q_columns.shape
    eta = parameter_features(mu, feature_mode=feature_mode)
    n_eta = eta.size
    q_t = q_columns.T
    quad = compact_quadratic_matrix(q_columns)
    n_quad = quad.shape[1]
    n_features = 1 + n_eta + n_red + n_quad
    if include_param_linear:
        n_features += n_eta * n_red
    if include_param_quad:
        n_features += n_eta * n_quad
    theta = np.empty((n_samples, n_features), dtype=np.float64)
    theta[:, 0] = 1.0
    theta[:, 1:1 + n_eta] = eta[None, :]
    q0 = 1 + n_eta
    q1 = q0 + n_red
    theta[:, q0:q1] = q_t
    col = q1
    if include_param_linear:
        for val in eta:
            theta[:, col:col + n_red] = val * q_t
            col += n_red
    theta[:, col:col + n_quad] = quad
    col += n_quad
    if include_param_quad:
        for val in eta:
            theta[:, col:col + n_quad] = val * quad
            col += n_quad
    return theta


def project_snapshots(snaps, basis, u_ref):
    """Project state snapshots into centered POD coordinates."""
    return _project_snapshots(snaps, basis, u_ref)


def reconstruct_snapshots(q_snaps, basis, u_ref):
    """Reconstruct full state snapshots from reduced coordinates."""
    return _reconstruct_snapshots(q_snaps, basis, u_ref)


def fit_linear_discrete_operator(theta, y_next, ridge=1e-8, penalize_intercept=False):
    """Fit q_{k+1} = W theta_k with column scaling and ridge regularization."""
    return fit_operator(
        theta,
        y_next,
        ridge=ridge,
        penalize_intercept=penalize_intercept,
        error_name="relative_one_step_error",
    )


def predict_next_q(q, mu, operator, x_mean, x_scale, feature_mode=FEATURE_MODE):
    theta = design_vector(q, mu, feature_mode=feature_mode)
    theta_scaled = (theta - x_mean) / x_scale
    return np.asarray(operator, dtype=np.float64) @ theta_scaled


def predict_next_q_quadratic(
    q,
    mu,
    operator,
    x_mean,
    x_scale,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_param_quad=True,
):
    theta = design_vector_quadratic(
        q,
        mu,
        feature_mode=feature_mode,
        include_param_linear=include_param_linear,
        include_param_quad=include_param_quad,
    )
    theta_scaled = (theta - x_mean) / x_scale
    return np.asarray(operator, dtype=np.float64) @ theta_scaled


def rollout_linear_discrete(
    q0,
    mu,
    num_steps,
    operator,
    x_mean,
    x_scale,
    feature_mode=FEATURE_MODE,
    max_norm=1e12,
):
    """Roll out the learned model and stop if it becomes numerically unstable."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    n_red = q0.size
    q_snaps = np.zeros((n_red, int(num_steps) + 1), dtype=np.float64)
    q_snaps[:, 0] = q0
    stable_steps = int(num_steps)
    unstable_reason = ""

    q = q0.copy()
    for istep in range(int(num_steps)):
        q = predict_next_q(q, mu, operator, x_mean, x_scale, feature_mode=feature_mode)
        if (not np.all(np.isfinite(q))) or np.linalg.norm(q) > float(max_norm):
            stable_steps = istep
            unstable_reason = f"unstable at step {istep + 1}"
            q_snaps[:, istep + 1:] = np.nan
            break
        q_snaps[:, istep + 1] = q

    return q_snaps, stable_steps, unstable_reason


def rollout_quadratic_discrete(
    q0,
    mu,
    num_steps,
    operator,
    x_mean,
    x_scale,
    feature_mode=FEATURE_MODE,
    max_norm=1e12,
    include_param_linear=True,
    include_param_quad=True,
):
    """Roll out the learned quadratic model and stop if it becomes numerically unstable."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    n_red = q0.size
    q_snaps = np.zeros((n_red, int(num_steps) + 1), dtype=np.float64)
    q_snaps[:, 0] = q0
    stable_steps = int(num_steps)
    unstable_reason = ""

    q = q0.copy()
    for istep in range(int(num_steps)):
        q = predict_next_q_quadratic(
            q,
            mu,
            operator,
            x_mean,
            x_scale,
            feature_mode=feature_mode,
            include_param_linear=include_param_linear,
            include_param_quad=include_param_quad,
        )
        if (not np.all(np.isfinite(q))) or np.linalg.norm(q) > float(max_norm):
            stable_steps = istep
            unstable_reason = f"unstable at step {istep + 1}"
            q_snaps[:, istep + 1:] = np.nan
            break
        q_snaps[:, istep + 1] = q

    return q_snaps, stable_steps, unstable_reason


def load_pod_data(pod_dir, state_size, num_modes=None):
    basis_path = os.path.join(pod_dir, "basis.npy")
    sigma_path = os.path.join(pod_dir, "sigma.npy")
    u_ref_path = os.path.join(pod_dir, "u_ref.npy")
    metadata_path = os.path.join(pod_dir, "stage1_pod_metadata.npz")

    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"POD basis not found at {basis_path}. Run POD/stage1_build_pod_basis.py first.")
    if not os.path.exists(sigma_path):
        raise FileNotFoundError(f"POD singular values not found at {sigma_path}.")

    basis_full = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    sigma = np.asarray(np.load(sigma_path, allow_pickle=False), dtype=np.float64)
    if basis_full.ndim != 2 or basis_full.shape[0] != state_size:
        raise ValueError(f"Basis/state mismatch: basis shape {basis_full.shape}, state size {state_size}.")

    if os.path.exists(u_ref_path):
        u_ref = np.asarray(np.load(u_ref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(state_size, dtype=np.float64)
    if u_ref.size != state_size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {state_size}.")

    n_available = basis_full.shape[1]
    n_keep = n_available if num_modes is None else int(num_modes)
    if n_keep < 1 or n_keep > n_available:
        raise ValueError(f"Requested num_modes={n_keep}, available modes={n_available}.")
    basis = basis_full[:, :n_keep]

    metadata = {}
    if os.path.exists(metadata_path):
        data = np.load(metadata_path, allow_pickle=True)
        for key in data.files:
            val = data[key]
            metadata[key] = val.item() if np.asarray(val).shape == () else val

    energy_captured = None
    energy_lost = None
    if sigma.size > 0 and n_keep <= sigma.size:
        sigma_sq = sigma**2
        total_energy = float(np.sum(sigma_sq))
        if total_energy > 0.0:
            energy_captured = float(np.sum(sigma_sq[:n_keep]) / total_energy)
            energy_lost = 1.0 - energy_captured

    return basis, sigma, u_ref, metadata, basis_path, energy_captured, energy_lost


def save_model(
    model_path,
    operator,
    x_mean,
    x_scale,
    num_modes,
    ridge,
    feature_mode,
    train_mu,
    train_relative_one_step_error,
    dt,
    num_steps,
    pod_basis_path,
    energy_captured=None,
    energy_lost=None,
    model_family=MODEL_FAMILY,
    quadratic_param_mode="",
):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    np.savez(
        model_path,
        model_family=np.asarray(str(model_family)),
        operator=np.asarray(operator, dtype=np.float64),
        x_mean=np.asarray(x_mean, dtype=np.float64),
        x_scale=np.asarray(x_scale, dtype=np.float64),
        num_modes=np.asarray(int(num_modes), dtype=np.int64),
        ridge=np.asarray(float(ridge), dtype=np.float64),
        feature_mode=np.asarray(str(feature_mode)),
        train_mu=np.asarray(train_mu, dtype=np.float64),
        train_relative_one_step_error=np.asarray(float(train_relative_one_step_error), dtype=np.float64),
        dt=np.asarray(float(dt), dtype=np.float64),
        num_steps=np.asarray(int(num_steps), dtype=np.int64),
        pod_basis_path=np.asarray(str(pod_basis_path)),
        num_features=np.asarray(int(np.asarray(operator).shape[1]), dtype=np.int64),
        quadratic_param_mode=np.asarray(str(quadratic_param_mode)),
        energy_captured=np.asarray(np.nan if energy_captured is None else float(energy_captured), dtype=np.float64),
        energy_lost=np.asarray(np.nan if energy_lost is None else float(energy_lost), dtype=np.float64),
    )


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"OpInf model not found at {model_path}. Run OpInf/stage1_fit_linear_opinf.py first."
        )
    data = np.load(model_path, allow_pickle=False)
    model = {key: data[key] for key in data.files}
    family = str(model["model_family"])
    if family not in (MODEL_FAMILY, QUADRATIC_MODEL_FAMILY):
        raise ValueError(
            f"Unsupported model_family={family!r}; expected {MODEL_FAMILY!r} or {QUADRATIC_MODEL_FAMILY!r}."
        )
    model["model_family"] = family
    model["operator"] = np.asarray(model["operator"], dtype=np.float64)
    model["x_mean"] = np.asarray(model["x_mean"], dtype=np.float64)
    model["x_scale"] = np.asarray(model["x_scale"], dtype=np.float64)
    model["num_modes"] = int(np.asarray(model["num_modes"]).item())
    model["ridge"] = float(np.asarray(model["ridge"]).item())
    model["feature_mode"] = str(np.asarray(model["feature_mode"]).item())
    model["train_relative_one_step_error"] = float(np.asarray(model["train_relative_one_step_error"]).item())
    model["dt"] = float(np.asarray(model["dt"]).item())
    model["num_steps"] = int(np.asarray(model["num_steps"]).item())
    model["pod_basis_path"] = str(np.asarray(model["pod_basis_path"]).item())
    if "num_features" in model:
        model["num_features"] = int(np.asarray(model["num_features"]).item())
    if "quadratic_param_mode" in model:
        model["quadratic_param_mode"] = str(np.asarray(model["quadratic_param_mode"]).item())
    return model
