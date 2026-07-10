"""Utilities for POD-based polynomial-manifold OpInf ROMs."""

import itertools
import os
import sys

import numpy as np

from opinf_utils import FEATURE_MODE, parameter_features


LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)

from opinf_lab.features import compact_monomial_matrix as _shared_compact_monomial_matrix
from opinf_lab.features import continuous_feature_matrix as _shared_continuous_feature_matrix
from opinf_lab.features import continuous_feature_vector as _shared_continuous_feature_vector
from opinf_lab.regression import fit_operator
from opinf_lab.time_integration import rollout_rk4


MANIFOLD_MODEL_FAMILY = "mpod_quadratic_manifold_continuous"


def elementwise_power_features(q, polynomial_order=2):
    """Return [q**2, q**3, ..., q**p] using the paper's elementwise embedding."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = int(polynomial_order)
    if p < 2:
        raise ValueError(f"polynomial_order must be at least 2, got {polynomial_order}.")
    return np.concatenate([q**degree for degree in range(2, p + 1)])


def elementwise_power_matrix(q_columns, polynomial_order=2):
    """Vectorized [q**2, ..., q**p] for reduced states stored as columns."""
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim != 2:
        raise ValueError(f"q_columns must be 2D, got shape {q_columns.shape}.")
    p = int(polynomial_order)
    if p < 2:
        raise ValueError(f"polynomial_order must be at least 2, got {polynomial_order}.")
    blocks = [(q_columns.T) ** degree for degree in range(2, p + 1)]
    return np.hstack(blocks)


def higher_monomial_exponents(num_modes, degree):
    """Return exponent rows for the induced MPOD operator features ghat(q).

    The decoder uses g_p(q) = [q**2; ...; q**p].  If the full-order dynamics
    are quadratic in the state, substituting u = V q + Vbar Xi g_p(q) into
    the vector field generates all products among q and g_p(q).  This function
    lists those induced monomials after the constant, linear, and quadratic
    reduced terms have already been included.
    """
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

    # Linear full-order terms acting on the manifold polynomial g_p(q).
    for power in range(3, p + 1):
        for i in range(r):
            exponents.add(tuple(unit(i, power)))

    # Quadratic interactions between V q and Vbar Xi g_p(q).
    for i in range(r):
        for j in range(r):
            for power in range(2, p + 1):
                row = unit(i, 1)
                row[j] += power
                if sum(row) >= 3:
                    exponents.add(tuple(row))

    # Quadratic interactions inside Vbar Xi g_p(q).
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
    """Evaluate monomials q**exponents row by row."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    exponents = np.asarray(exponents, dtype=np.int64)
    if exponents.size == 0:
        return np.empty(0, dtype=np.float64)
    if exponents.ndim != 2 or exponents.shape[1] != q.size:
        raise ValueError(f"Exponent shape {exponents.shape} is incompatible with q size {q.size}.")
    values = np.ones(exponents.shape[0], dtype=np.float64)
    for j in range(q.size):
        powers = exponents[:, j]
        active = powers != 0
        if np.any(active):
            values[active] *= q[j] ** powers[active]
    return values


def induced_higher_feature_matrix(q_columns, exponents):
    """Evaluate the induced higher MPOD features for states stored as columns."""
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim != 2:
        raise ValueError(f"q_columns must be 2D, got shape {q_columns.shape}.")
    exponents = np.asarray(exponents, dtype=np.int64)
    if exponents.size == 0:
        return np.empty((q_columns.shape[1], 0), dtype=np.float64)
    return np.vstack([evaluate_monomials(q_columns[:, j], exponents) for j in range(q_columns.shape[1])])


def fit_polynomial_manifold(q_primary, q_secondary, polynomial_order=2, ridge=1e-8):
    """Fit q_secondary ~= Xi g(q_primary) with ridge regularization."""
    q_primary = np.asarray(q_primary, dtype=np.float64)
    q_secondary = np.asarray(q_secondary, dtype=np.float64)
    if q_primary.ndim != 2 or q_secondary.ndim != 2:
        raise ValueError("q_primary and q_secondary must be 2D arrays.")
    if q_primary.shape[1] != q_secondary.shape[1]:
        raise ValueError(f"Sample mismatch: primary {q_primary.shape}, secondary {q_secondary.shape}.")

    G = elementwise_power_matrix(q_primary, polynomial_order=polynomial_order)
    target = q_secondary.T
    reg = float(ridge)
    lhs = G.T @ G
    if reg > 0.0:
        lhs = lhs + reg * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = G.T @ target
    xi_t = np.linalg.solve(lhs, rhs)
    xi = xi_t.T
    pred = xi @ G.T
    denom = np.linalg.norm(q_secondary)
    rel_error = np.linalg.norm(q_secondary - pred) / (denom if denom > 0.0 else 1.0)
    return xi, float(rel_error)


def manifold_decode(q_primary, basis_primary, basis_secondary, xi, u_ref, polynomial_order=2):
    """Decode primary coordinates through u = u_ref + V q + W Xi g(q)."""
    q_primary = np.asarray(q_primary, dtype=np.float64)
    if q_primary.ndim == 1:
        q_primary = q_primary.reshape(-1, 1)
    basis_primary = np.asarray(basis_primary, dtype=np.float64)
    basis_secondary = np.asarray(basis_secondary, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    g = elementwise_power_matrix(q_primary, polynomial_order=polynomial_order).T
    q_secondary = xi @ g
    return u_ref[:, None] + basis_primary @ q_primary + basis_secondary @ q_secondary


def estimate_time_derivative(q_columns, dt):
    """Second-order finite-difference derivative estimate for q(t)."""
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim != 2:
        raise ValueError(f"q_columns must be 2D, got shape {q_columns.shape}.")
    if q_columns.shape[1] < 3:
        raise ValueError("At least three time samples are required for derivative estimation.")
    return np.gradient(q_columns, float(dt), axis=1, edge_order=2)


def _monomial_combinations(num_vars, degree):
    return np.asarray(list(itertools.combinations_with_replacement(range(int(num_vars)), int(degree))), dtype=np.int64)


def compact_monomial_matrix(q_columns, degree):
    """All unique monomials of a given total degree for states stored as columns."""
    return _shared_compact_monomial_matrix(q_columns, degree=degree)


def continuous_feature_matrix(
    q_columns,
    mu,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_quadratic=True,
    include_higher=True,
    max_degree=4,
):
    """Build features for q_dot = W theta(q, mu)."""
    eta = parameter_features(mu, feature_mode=feature_mode)
    return _shared_continuous_feature_matrix(
        q_columns,
        parameter_features=eta,
        include_constant=True,
        include_parameters=True,
        include_state=True,
        include_param_state=include_param_linear,
        include_quadratic=include_quadratic,
        include_param_quadratic=False,
        include_higher=include_higher,
        max_degree=max_degree,
    )


def continuous_feature_vector(
    q,
    mu,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_quadratic=True,
    include_higher=True,
    max_degree=4,
):
    eta = parameter_features(mu, feature_mode=feature_mode)
    return _shared_continuous_feature_vector(
        q,
        parameter_features=eta,
        include_constant=True,
        include_parameters=True,
        include_state=True,
        include_param_state=include_param_linear,
        include_quadratic=include_quadratic,
        include_param_quadratic=False,
        include_higher=include_higher,
        max_degree=max_degree,
    )


def manifold_continuous_feature_matrix(
    q_columns,
    mu,
    polynomial_order,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_quadratic=True,
    include_higher=False,
    max_degree=4,
    include_manifold_dynamics=True,
    exponents=None,
):
    """Build MPOD dynamics features with the induced higher library ghat(q)."""
    base = continuous_feature_matrix(
        q_columns,
        mu,
        feature_mode=feature_mode,
        include_param_linear=include_param_linear,
        include_quadratic=include_quadratic,
        include_higher=include_higher,
        max_degree=max_degree,
    )
    if not include_manifold_dynamics:
        return base
    if exponents is None:
        exponents = higher_monomial_exponents(q_columns.shape[0], polynomial_order)
    extra = induced_higher_feature_matrix(q_columns, exponents)
    return np.hstack((base, extra))


def manifold_continuous_feature_vector(q, mu, polynomial_order, **kwargs):
    q = np.asarray(q, dtype=np.float64).reshape(-1, 1)
    return manifold_continuous_feature_matrix(q, mu, polynomial_order=polynomial_order, **kwargs)[0]


def fit_continuous_operator(theta, qdot, ridge=1e-6, penalize_intercept=False):
    """Fit q_dot = W theta with column scaling and ridge regularization."""
    return fit_operator(
        theta,
        qdot,
        ridge=ridge,
        penalize_intercept=penalize_intercept,
        error_name="relative_derivative_training_error",
    )


def rhs_continuous(q, mu, model):
    include_manifold_dynamics = bool(model.get("include_manifold_dynamics", False))
    if include_manifold_dynamics:
        operator_library = model.get("manifold_operator_library", None)
        if operator_library != "induced_higher":
            raise ValueError(
                "This MPOD implementation only supports the induced_higher operator library. "
                "Retrain with OpInf/stage1_fit_manifold_opinf.py."
            )
        exponents = model.get("exponents", None)
        if exponents is None:
            raise ValueError("Model is missing induced higher monomial exponents; retrain the MPOD model.")
        theta = manifold_continuous_feature_vector(
            q,
            mu,
            polynomial_order=int(model["polynomial_order"]),
            feature_mode=model["feature_mode"],
            include_param_linear=bool(model["include_param_linear"]),
            include_quadratic=bool(model["include_quadratic"]),
            include_higher=bool(model["include_higher"]),
            max_degree=int(model["max_degree"]),
            include_manifold_dynamics=True,
            exponents=exponents,
        )
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


def rollout_continuous_rk4(q0, mu, dt, num_steps, model, max_norm=1e12, substeps=1):
    """Fixed-step RK4 rollout for the inferred continuous-time reduced model."""
    return rollout_rk4(
        lambda q: rhs_continuous(q, mu, model),
        q0,
        dt,
        num_steps,
        max_norm=max_norm,
        substeps=substeps,
    )


def save_manifold_model(model_path, **kwargs):
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
    arrays["model_family"] = np.asarray(MANIFOLD_MODEL_FAMILY)
    np.savez(model_path, **arrays)


def load_manifold_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Manifold OpInf model not found at {model_path}. Run OpInf/stage1_fit_manifold_opinf.py first."
        )
    data = np.load(model_path, allow_pickle=False)
    model = {key: data[key] for key in data.files}
    family = str(np.asarray(model["model_family"]).item())
    if family != MANIFOLD_MODEL_FAMILY:
        raise ValueError(f"Unsupported model_family={family!r}; expected {MANIFOLD_MODEL_FAMILY!r}.")
    for key in (
        "feature_mode",
        "pod_basis_path",
        "manifold_feature_type",
        "manifold_operator_library",
        "dynamics_feature_type",
    ):
        if key in model:
            model[key] = str(np.asarray(model[key]).item())
    for key in (
        "num_primary",
        "num_secondary",
        "polynomial_order",
        "num_steps",
        "max_degree",
        "num_features",
        "num_higher_features",
    ):
        if key in model:
            model[key] = int(np.asarray(model[key]).item())
    for key in (
        "dt",
        "manifold_ridge",
        "dynamics_ridge",
        "relative_manifold_training_error",
        "relative_derivative_training_error",
        "energy_captured_primary",
        "energy_captured_total_basis",
    ):
        if key in model:
            model[key] = float(np.asarray(model[key]).item())
    for key in ("include_param_linear", "include_quadratic", "include_higher", "include_manifold_dynamics"):
        if key in model:
            model[key] = bool(int(np.asarray(model[key]).item()))
    model["operator"] = np.asarray(model["operator"], dtype=np.float64)
    model["x_mean"] = np.asarray(model["x_mean"], dtype=np.float64)
    model["x_scale"] = np.asarray(model["x_scale"], dtype=np.float64)
    model["xi"] = np.asarray(model["xi"], dtype=np.float64)
    if "exponents" in model:
        model["exponents"] = np.asarray(model["exponents"], dtype=np.int64)
    return model
