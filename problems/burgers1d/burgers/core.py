"""Core 1D Burgers HDM, POD, IO, and plotting utilities."""

import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from .gauss_newton import newton_raphson


def make_1d_grid(x_low, x_up, num_cells):
    return np.linspace(float(x_low), float(x_up), int(num_cells) + 1)


def make_ddx(grid_x):
    grid_x = np.asarray(grid_x, dtype=np.float64)
    dx = grid_x[1:] - grid_x[:-1]
    return sp.spdiags(
        [-np.ones(dx.size) / dx, np.ones(dx.size) / dx],
        [-1, 0],
        dx.size,
        dx.size,
        format="csr",
    )


def get_ops(grid_x):
    Dxec = make_ddx(grid_x)
    Eye = sp.identity(grid_x.size - 1, format="csr")
    return Dxec, Eye


def source_term(grid_x, mu2):
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])
    return 0.02 * np.exp(float(mu2) * xc)


def inviscid_burgers_res1d(u, grid_x, dt, up, mu, Dxec=None):
    """Backward-Euler finite-volume residual with positive upwind/Godunov flux."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    up = np.asarray(up, dtype=np.float64).reshape(-1)
    if Dxec is None:
        Dxec = make_ddx(grid_x)

    dx = grid_x[1:] - grid_x[:-1]
    flux = 0.5 * u**2
    src = float(dt) * source_term(grid_x, mu[1])

    res = u - up + float(dt) * (Dxec @ flux) - src
    res[0] -= 0.5 * float(dt) * float(mu[0]) ** 2 / dx[0]
    return res


def inviscid_burgers_exact_jac1d(u, dt, Dxec=None, Eye=None, grid_x=None):
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if Dxec is None:
        if grid_x is None:
            raise ValueError("Either Dxec or grid_x must be provided.")
        Dxec = make_ddx(grid_x)
    if Eye is None:
        Eye = sp.identity(u.size, format="csr")
    return Eye + float(dt) * (Dxec @ sp.diags(u, format="csr"))


def inviscid_burgers_res1d_ecsw(
    u_aug,
    grid_x,
    dt,
    up_aug,
    mu,
    sample_inds,
    augmented_inds,
):
    """Backward-Euler residual restricted to sampled 1D cells."""
    u_aug = np.asarray(u_aug, dtype=np.float64).reshape(-1)
    up_aug = np.asarray(up_aug, dtype=np.float64).reshape(-1)
    sample_inds = np.asarray(sample_inds, dtype=int).reshape(-1)
    augmented_inds = np.asarray(augmented_inds, dtype=int).reshape(-1)

    pos = {int(idx): i for i, idx in enumerate(augmented_inds)}
    dx = grid_x[1:] - grid_x[:-1]
    src = source_term(grid_x, mu[1])
    res = np.zeros(sample_inds.size, dtype=np.float64)

    for row, cell in enumerate(sample_inds):
        cell = int(cell)
        p_i = pos[cell]
        u_i = u_aug[p_i]
        up_i = up_aug[p_i]
        flux_i = 0.5 * u_i**2

        if cell == 0:
            flux_left = 0.5 * float(mu[0]) ** 2
        else:
            p_l = pos[cell - 1]
            flux_left = 0.5 * u_aug[p_l] ** 2

        res[row] = (
            u_i
            - up_i
            + float(dt) / dx[cell] * (flux_i - flux_left)
            - float(dt) * src[cell]
        )

    return res


def inviscid_burgers_exact_jac1d_ecsw(
    u_aug,
    dt,
    grid_x,
    sample_inds,
    augmented_inds,
):
    """Sparse sampled Jacobian dR_sample / du_aug for 1D ECSW."""
    u_aug = np.asarray(u_aug, dtype=np.float64).reshape(-1)
    sample_inds = np.asarray(sample_inds, dtype=int).reshape(-1)
    augmented_inds = np.asarray(augmented_inds, dtype=int).reshape(-1)

    pos = {int(idx): i for i, idx in enumerate(augmented_inds)}
    dx = grid_x[1:] - grid_x[:-1]

    rows = []
    cols = []
    data = []

    for row, cell in enumerate(sample_inds):
        cell = int(cell)
        p_i = pos[cell]
        rows.append(row)
        cols.append(p_i)
        data.append(1.0 + float(dt) * u_aug[p_i] / dx[cell])

        if cell > 0:
            p_l = pos[cell - 1]
            rows.append(row)
            cols.append(p_l)
            data.append(-float(dt) * u_aug[p_l] / dx[cell])

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(sample_inds.size, augmented_inds.size),
    )


def inviscid_burgers_implicit1d(
    grid_x,
    u0,
    dt,
    num_steps,
    mu,
    max_its=30,
    relnorm_cutoff=1e-10,
    verbose=True,
):
    u0 = np.asarray(u0, dtype=np.float64).reshape(-1)
    if verbose:
        print(
            "Running 1D HDM "
            f"(backward-Euler FV) for mu1={float(mu[0]):.3f}, mu2={float(mu[1]):.4f}"
        )

    snaps = np.zeros((u0.size, int(num_steps) + 1), dtype=np.float64)
    snaps[:, 0] = u0.copy()
    up = u0.copy()

    Dxec, Eye = get_ops(grid_x)
    for istep in range(int(num_steps)):
        if verbose:
            print(f"  HDM timestep {istep + 1}/{num_steps}")

        def res(u):
            return inviscid_burgers_res1d(u, grid_x, dt, up, mu, Dxec)

        def jac(u):
            return inviscid_burgers_exact_jac1d(u, dt, Dxec, Eye)

        u, _ = newton_raphson(
            func=res,
            jac=jac,
            x0=up,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            verbose=verbose,
        )
        snaps[:, istep + 1] = u
        up = u.copy()

    return snaps


def get_snapshot_params():
    from .config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

    mu1 = np.linspace(MU1_RANGE[0], MU1_RANGE[1], SAMPLES_PER_MU)
    mu2 = np.linspace(MU2_RANGE[0], MU2_RANGE[1], SAMPLES_PER_MU)
    return [[float(a), float(b)] for a in mu1 for b in mu2]


def param_to_snap_fn(mu, snap_folder="param_snaps"):
    return os.path.join(
        snap_folder,
        f"snaps_be_mu1_{float(mu[0]):.3f}_mu2_{float(mu[1]):.4f}.npy",
    )


def load_or_compute_snaps(
    mu,
    grid_x,
    u0,
    dt,
    num_steps,
    snap_folder="param_snaps",
    force=False,
    verbose=True,
):
    os.makedirs(snap_folder, exist_ok=True)
    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    if os.path.exists(snap_fn) and not force:
        snaps = np.load(snap_fn, allow_pickle=False)
        expected_shape = (np.asarray(u0).size, int(num_steps) + 1)
        if snaps.shape == expected_shape:
            return snaps
        if verbose:
            print(
                f"Cached snapshots at {snap_fn} have shape {snaps.shape}, "
                f"expected {expected_shape}; recomputing."
            )

    t0 = time.time()
    snaps = inviscid_burgers_implicit1d(grid_x, u0, dt, num_steps, mu, verbose=verbose)
    np.save(snap_fn, snaps)
    if verbose:
        print(f"Saved snapshots to {snap_fn} in {time.time() - t0:.3e}s")
    return snaps


def POD(
    snaps,
    num_modes=None,
    method="svd",
    random_state=None,
    energy_capture=None,
    energy_loss=None,
    min_size=None,
    max_size=None,
    return_truncation_info=False,
    center=False,
    u_ref=None,
    return_reference=False,
):
    snaps = np.asarray(snaps, dtype=np.float64)
    n_dofs = snaps.shape[0]

    u_ref_vec = None
    reference_source = "none"
    if center or u_ref is not None:
        if u_ref is None:
            u_ref_vec = np.mean(snaps, axis=1)
            reference_source = "mean"
        else:
            u_ref_vec = np.asarray(u_ref, dtype=np.float64).reshape(-1)
            if u_ref_vec.size != n_dofs:
                raise ValueError(f"u_ref has size {u_ref_vec.size}, expected {n_dofs}")
            reference_source = "provided"

    snaps_pod = snaps if u_ref_vec is None else snaps - u_ref_vec[:, None]

    if energy_capture is not None and energy_loss is not None:
        raise ValueError("Specify only one of energy_capture or energy_loss.")
    if energy_loss is not None:
        if not (0.0 <= energy_loss <= 1.0):
            raise ValueError("energy_loss must lie in [0, 1].")
        energy_capture = 1.0 - energy_loss
    if energy_capture is not None and not (0.0 <= energy_capture <= 1.0):
        raise ValueError("energy_capture must lie in [0, 1].")

    if method == "svd":
        u, s, _ = np.linalg.svd(snaps_pod, full_matrices=False)
    elif method == "rsvd":
        if num_modes is None:
            raise ValueError("For method='rsvd', num_modes must be provided.")
        from sklearn.utils.extmath import randomized_svd

        u, s, _ = randomized_svd(
            snaps_pod,
            n_components=int(num_modes),
            random_state=random_state,
        )
    else:
        raise ValueError("method must be 'svd' or 'rsvd'.")

    if num_modes is None:
        if energy_capture is None:
            n_keep = s.size
        else:
            total = float(np.sum(s**2))
            cumulative = np.cumsum(s**2) / total if total > 0.0 else np.ones_like(s)
            n_keep = int(np.searchsorted(cumulative, energy_capture, side="left") + 1)
    else:
        n_keep = int(num_modes)

    if min_size is not None:
        n_keep = max(n_keep, int(min_size))
    if max_size is not None:
        n_keep = min(n_keep, int(max_size))
    n_keep = max(1, min(n_keep, u.shape[1]))

    basis = u[:, :n_keep]
    total_energy = float(np.sum(s**2))
    captured = float(np.sum(s[:n_keep] ** 2) / total_energy) if total_energy > 0.0 else 1.0
    info = {
        "n_keep": n_keep,
        "n_available": int(s.size),
        "energy_captured": captured,
        "energy_lost": 1.0 - captured,
        "centered": bool(u_ref_vec is not None),
        "reference_source": reference_source,
    }

    outputs = [basis, s]
    if return_truncation_info:
        outputs.append(info)
    if return_reference:
        outputs.append(u_ref_vec)
    return tuple(outputs) if len(outputs) > 2 else (basis, s)


def plot_singular_value_decay(
    sigma,
    out_path,
    max_modes=200,
    label="POD",
    title="POD residual energy decay",
    use_latex=False,
):
    del use_latex
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0:
        raise ValueError("sigma is empty.")
    energy = sigma**2
    total = float(np.sum(energy))
    residual = 1.0 - np.cumsum(energy) / total if total > 0.0 else np.zeros_like(energy)
    nmodes = min(int(max_modes), sigma.size)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, nmodes + 1), np.maximum(residual[:nmodes], 1e-18), label=label)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Discarded energy")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_snaps(grid_x, hdm_snaps, rom_snaps=None, steps=None, out_path=None, title=None):
    x = 0.5 * (grid_x[1:] + grid_x[:-1])
    hdm_snaps = np.asarray(hdm_snaps, dtype=np.float64)
    if steps is None:
        steps = np.linspace(0, hdm_snaps.shape[1] - 1, 6, dtype=int)

    fig, ax = plt.subplots(figsize=(9, 5))
    for step in steps:
        ax.plot(x, hdm_snaps[:, step], color="black", alpha=0.35, linewidth=1.8)
        if rom_snaps is not None:
            ax.plot(x, rom_snaps[:, step], "--", alpha=0.85, linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title or "1D Burgers snapshots")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig, ax
