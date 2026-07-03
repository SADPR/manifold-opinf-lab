"""Feature construction shared by OpInf examples.

The functions here are intentionally equation-agnostic. They operate on reduced
coordinates q and optional parameter features eta(mu), but they do not know how
snapshots were generated or what PDE is being approximated.
"""

import itertools

import numpy as np


def _as_columns(q_columns):
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim == 1:
        q_columns = q_columns.reshape(-1, 1)
    if q_columns.ndim != 2:
        raise ValueError(f"Expected a 2D column array, got shape {q_columns.shape}.")
    return q_columns


def polynomial_parameter_features(mu, degree=2, include_constant=False):
    """Return all parameter monomials up to ``degree``.

    For the current two-parameter Burgers case and ``degree=2`` this returns
    ``[mu1, mu2, mu1**2, mu1*mu2, mu2**2]``, matching the previous scripts.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    max_degree = int(degree)
    if max_degree < 0:
        raise ValueError("degree must be nonnegative.")

    values = []
    if include_constant:
        values.append(1.0)
    for deg in range(1, max_degree + 1):
        for combo in itertools.combinations_with_replacement(range(mu.size), deg):
            values.append(float(np.prod(mu[list(combo)])))
    return np.asarray(values, dtype=np.float64)


def compact_monomials(q, degree=2):
    """Return compact monomials of a single reduced state.

    ``degree=2`` gives the upper-triangular products ``q_i q_j`` with
    ``i <= j``.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    deg = int(degree)
    if deg < 1:
        raise ValueError("degree must be at least one.")
    combos = list(itertools.combinations_with_replacement(range(q.size), deg))
    out = np.empty(len(combos), dtype=np.float64)
    for i, combo in enumerate(combos):
        out[i] = np.prod(q[list(combo)])
    return out


def compact_monomial_matrix(q_columns, degree=2):
    """Vectorized compact monomials for reduced states stored as columns."""
    q_columns = _as_columns(q_columns)
    deg = int(degree)
    if deg < 1:
        raise ValueError("degree must be at least one.")
    combos = np.asarray(
        list(itertools.combinations_with_replacement(range(q_columns.shape[0]), deg)),
        dtype=np.int64,
    )
    if combos.size == 0:
        return np.zeros((q_columns.shape[1], 0), dtype=np.float64)
    out = np.ones((q_columns.shape[1], combos.shape[0]), dtype=np.float64)
    for j in range(combos.shape[1]):
        out *= q_columns[combos[:, j], :].T
    return out


def continuous_feature_matrix(
    q_columns,
    parameter_features=None,
    include_constant=True,
    include_parameters=True,
    include_state=True,
    include_param_state=False,
    include_quadratic=True,
    include_param_quadratic=False,
    include_higher=False,
    max_degree=4,
    extra_blocks=(),
    include_param_extra=False,
):
    """Build a generic continuous-time OpInf feature matrix.

    The returned rows are samples and columns are feature components. With
    parameter features ``eta`` and an extra nonlinear-map block ``z(q, mu)``,
    the common Burgers ANN-NM-MPOD case is

    ``[1, eta, q, eta*q, q_quad, z]``.
    """
    q_columns = _as_columns(q_columns)
    q_t = q_columns.T
    n_samples = q_t.shape[0]
    eta = np.asarray([], dtype=np.float64)
    if parameter_features is not None:
        eta = np.asarray(parameter_features, dtype=np.float64).reshape(-1)

    blocks = []
    if include_constant:
        blocks.append(np.ones((n_samples, 1), dtype=np.float64))
    if include_parameters and eta.size:
        blocks.append(np.tile(eta, (n_samples, 1)))
    if include_state:
        blocks.append(q_t)
    if include_param_state and eta.size:
        blocks.extend([float(val) * q_t for val in eta])
    if include_quadratic:
        q_quad = compact_monomial_matrix(q_columns, degree=2)
        blocks.append(q_quad)
        if include_param_quadratic and eta.size:
            blocks.extend([float(val) * q_quad for val in eta])
    if include_higher:
        for degree in range(3, int(max_degree) + 1):
            blocks.append(compact_monomial_matrix(q_columns, degree=degree))

    for block in extra_blocks or ():
        arr = np.asarray(block, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != n_samples:
            raise ValueError(f"Extra feature block has {arr.shape[0]} rows, expected {n_samples}.")
        blocks.append(arr)
        if include_param_extra and eta.size:
            blocks.extend([float(val) * arr for val in eta])

    if not blocks:
        return np.zeros((n_samples, 0), dtype=np.float64)
    return np.hstack(blocks)


def continuous_feature_vector(q, **kwargs):
    """Single-state wrapper around :func:`continuous_feature_matrix`."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 1)
    return continuous_feature_matrix(q, **kwargs)[0]
