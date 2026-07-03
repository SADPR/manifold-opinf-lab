"""Newton and Gauss-Newton solvers used by the HDM and PROM."""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _safe_init_norm(value):
    return 1.0 if value == 0.0 else value


def _relative_drop(prev, curr):
    if prev == 0.0:
        return 0.0
    return abs((prev - curr) / prev)


def _prepare_u_ref(u_ref, size):
    if u_ref is None:
        return np.zeros(size, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if u_ref.size != size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {size}")
    return u_ref


def _solve_reduced_update(JV, r, linear_solver="lstsq", normal_eq_reg=1e-12):
    mode = str(linear_solver).strip().lower()
    if mode == "lstsq":
        dy, *_ = np.linalg.lstsq(JV, -r, rcond=None)
        return dy

    if mode == "normal_eq":
        reg = float(normal_eq_reg)
        if reg < 0.0:
            raise ValueError(f"normal_eq_reg must be non-negative, got {reg}.")
        ata = JV.T @ JV
        atb = -(JV.T @ r)
        if reg > 0.0:
            ata = ata + reg * np.eye(ata.shape[0], dtype=ata.dtype)
        try:
            return np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            dy, *_ = np.linalg.lstsq(JV, -r, rcond=None)
            return dy

    raise ValueError("linear_solver must be 'lstsq' or 'normal_eq'.")


def newton_raphson(func, jac, x0, max_its=30, relnorm_cutoff=1e-10, verbose=True):
    x = np.asarray(x0, dtype=np.float64).copy()
    r = func(x)
    init_norm = _safe_init_norm(np.linalg.norm(r))
    resnorms = []

    for it in range(max_its):
        if it > 0:
            r = func(x)
        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)
        relnorm = resnorm / init_norm
        if verbose:
            print(f"    Newton {it:02d}: relative residual {relnorm:.3e}")
        if relnorm < relnorm_cutoff:
            break

        J = jac(x)
        if sp.issparse(J):
            dx = spla.spsolve(J.tocsc(), r)
        else:
            dx = np.linalg.solve(J, r)
        x -= dx

    return x, resnorms


def gauss_newton_lspg(
    func,
    jac,
    basis,
    y0,
    max_its=20,
    relnorm_cutoff=1e-5,
    optimality_cutoff=1e-8,
    step_cutoff=1e-10,
    min_delta=1e-2,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    basis = np.asarray(basis, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64).copy()
    u_ref = _prepare_u_ref(u_ref, basis.shape[0])
    u = u_ref + basis @ y

    r = func(u)
    init_norm = _safe_init_norm(np.linalg.norm(r))
    init_opt_norm = None
    resnorms = []
    last_opt_norm = np.nan
    last_rel_opt = np.nan
    last_rel_step = np.nan
    reason = "max_iterations"

    for it in range(max_its):
        if it > 0:
            t0 = time.time()
            r = func(u)
            res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)
        relnorm = resnorm / init_norm
        if relnorm < relnorm_cutoff:
            reason = "full_residual"
            break

        t0 = time.time()
        J = jac(u)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ basis
        grad = JV.T @ r
        last_opt_norm = np.linalg.norm(grad)
        if init_opt_norm is None:
            init_opt_norm = _safe_init_norm(last_opt_norm)
        last_rel_opt = last_opt_norm / init_opt_norm
        if last_rel_opt < optimality_cutoff:
            reason = "lspg_stationarity"
            ls_time += time.time() - t0
            break

        dy = _solve_reduced_update(
            JV,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        last_rel_step = np.linalg.norm(dy) / (np.linalg.norm(y) + 1.0)
        if last_rel_step < step_cutoff:
            reason = "small_reduced_step"
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            reason = "full_residual_stagnation"
            break

        y += dy
        u = u_ref + basis @ y

    print(
        f"    GN {it:02d}: rel_full_res={resnorm / init_norm:.3e}, "
        f"rel_lspg_opt={last_rel_opt:.3e}, rel_step={last_rel_step:.3e}, "
        f"reason={reason}"
    )
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_lspg_weighted(
    func,
    jac,
    basis,
    y0,
    sample_weights,
    max_its=20,
    relnorm_cutoff=1e-5,
    optimality_cutoff=1e-8,
    step_cutoff=1e-10,
    min_delta=1e-2,
    u_ref=None,
    linear_solver="normal_eq",
    normal_eq_reg=1e-12,
):
    """Weighted Gauss-Newton for sampled ECSW LSPG systems."""
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    basis = np.asarray(basis, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64).copy()
    u_ref = _prepare_u_ref(u_ref, basis.shape[0])
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if np.any(weights < 0.0):
        raise ValueError("sample_weights must be non-negative.")
    sqrt_w = np.sqrt(weights)

    u = u_ref + basis @ y
    r = func(u)
    if r.size != weights.size:
        raise ValueError(f"Residual/weights size mismatch: {r.size} vs {weights.size}.")

    init_norm = _safe_init_norm(np.linalg.norm(sqrt_w * r))
    init_opt_norm = None
    resnorms = []
    last_rel_opt = np.nan
    last_rel_step = np.nan
    reason = "max_iterations"

    for it in range(max_its):
        if it > 0:
            t0 = time.time()
            r = func(u)
            res_time += time.time() - t0

        wr = sqrt_w * r
        resnorm = np.linalg.norm(wr)
        resnorms.append(resnorm)
        relnorm = resnorm / init_norm
        if relnorm < relnorm_cutoff:
            reason = "weighted_residual"
            break

        t0 = time.time()
        J = jac(u)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ basis
        JVw = sqrt_w[:, None] * JV
        grad = JVw.T @ wr
        opt_norm = np.linalg.norm(grad)
        if init_opt_norm is None:
            init_opt_norm = _safe_init_norm(opt_norm)
        last_rel_opt = opt_norm / init_opt_norm
        if last_rel_opt < optimality_cutoff:
            reason = "weighted_lspg_stationarity"
            ls_time += time.time() - t0
            break

        dy = _solve_reduced_update(
            JVw,
            wr,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        last_rel_step = np.linalg.norm(dy) / (np.linalg.norm(y) + 1.0)
        if last_rel_step < step_cutoff:
            reason = "small_reduced_step"
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            reason = "weighted_residual_stagnation"
            break

        y += dy
        u = u_ref + basis @ y

    print(
        f"    ECSW GN {it:02d}: rel_weighted_res={resnorm / init_norm:.3e}, "
        f"rel_lspg_opt={last_rel_opt:.3e}, rel_step={last_rel_step:.3e}, "
        f"reason={reason}"
    )
    return y, resnorms, (jac_time, res_time, ls_time)
