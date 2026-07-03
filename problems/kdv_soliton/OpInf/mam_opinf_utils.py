"""Utilities for alternating-minimization manifold OpInf on KdV."""

import os

import numpy as np
from scipy import optimize

from mpod_opinf_utils import (
    compute_mpod_manifold,
    polynomial_embedding,
    reconstruct_mpod_snapshots,
)


MODEL_FAMILY = "kdv_mam_polynomial_continuous_opinf"


def polynomial_jacobian(q, degree):
    """Return d/dq [q**2; ...; q**p] for one reduced coordinate vector."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    blocks = []
    for power in range(2, int(degree) + 1):
        blocks.append(np.diag(float(power) * q ** (power - 1)))
    return np.vstack(blocks)


def _xi_normal_equation(basis, basis_bar, centered, q, degree, gamma):
    """Update Xi with V, Vbar, and q fixed."""
    poly = polynomial_embedding(q, int(degree))
    projection_error = centered - basis @ q
    normal = poly @ poly.T + float(gamma) * np.eye(poly.shape[0], dtype=np.float64)
    rhs = basis_bar.T @ projection_error @ poly.T
    xi = np.linalg.solve(normal, rhs.T).T
    return xi, poly


def _solve_centered_coordinates(
    centered_snapshot,
    initial_guess,
    basis,
    basis_bar_xi,
    xi,
    degree,
    ls_ftol,
    ls_max_nfev=None,
):
    """Solve one nonlinear least-squares coordinate problem for MAM."""
    centered_snapshot = np.asarray(centered_snapshot, dtype=np.float64).reshape(-1)
    initial_guess = np.asarray(initial_guess, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis_bar_xi = np.asarray(basis_bar_xi, dtype=np.float64)

    def residual(q):
        return centered_snapshot - basis @ q - basis_bar_xi @ polynomial_embedding(q, int(degree))

    def jacobian(q):
        return -basis - basis_bar_xi @ polynomial_jacobian(q, int(degree))

    return optimize.least_squares(
        residual,
        initial_guess,
        jac=jacobian,
        ftol=float(ls_ftol),
        xtol=float(ls_ftol),
        gtol=float(ls_ftol),
        max_nfev=ls_max_nfev,
    )


def solve_mam_coordinates(
    snapshot,
    initial_guess,
    basis,
    basis_bar,
    xi,
    u_ref,
    degree,
    ls_ftol=1e-9,
    ls_max_nfev=None,
):
    """Compute MAM coordinates for one full-order snapshot."""
    centered_snapshot = np.asarray(snapshot, dtype=np.float64) - np.asarray(u_ref, dtype=np.float64)
    basis_bar_xi = np.asarray(basis_bar, dtype=np.float64) @ np.asarray(xi, dtype=np.float64)
    result = _solve_centered_coordinates(
        centered_snapshot,
        initial_guess,
        basis,
        basis_bar_xi,
        xi,
        degree,
        ls_ftol,
        ls_max_nfev=ls_max_nfev,
    )
    return result.x, result


def reconstruct_mam_snapshots(q, basis, basis_bar, xi, u_ref, degree):
    """Reconstruct full states from MAM coordinates."""
    return reconstruct_mpod_snapshots(q, basis, basis_bar, xi, u_ref, degree)


def compute_mam_manifold(
    train_snapshots,
    num_modes,
    total_modes,
    degree,
    gamma,
    tol=1e-3,
    max_iter=100,
    ls_ftol=1e-9,
    ls_max_nfev=None,
    coordinate_initialization="same_snapshot",
    progress=True,
):
    """Compute Algorithm-2 alternating-minimization manifold data."""
    train_snapshots = np.asarray(train_snapshots, dtype=np.float64)
    r = int(num_modes)
    total = int(total_modes)
    p = int(degree)
    q_secondary = total - r
    if q_secondary < 1:
        raise ValueError("total_modes must be larger than num_modes.")
    if coordinate_initialization not in {"same_snapshot", "previous_time"}:
        raise ValueError("coordinate_initialization must be 'same_snapshot' or 'previous_time'.")

    initial = compute_mpod_manifold(
        train_snapshots,
        num_modes=r,
        total_modes=total,
        degree=p,
        gamma=float(gamma),
    )
    u_ref = initial["u_ref"]
    centered = train_snapshots - u_ref[:, None]
    denom = max(np.linalg.norm(centered, "fro"), 1e-15)
    basis = initial["basis"].copy()
    basis_bar = initial["basis_bar"].copy()
    sigma = initial["sigma"]
    q = initial["q"].copy()
    xi = initial["xi"].copy()
    poly = polynomial_embedding(q, p)
    pod_basis = np.hstack([basis, basis_bar])

    def metrics(iteration, diff, nfev_total, nfev_max):
        reconstruction = reconstruct_mam_snapshots(q, basis, basis_bar, xi, u_ref, p)
        relative_reconstruction_error = float(np.linalg.norm(train_snapshots - reconstruction, "fro") / denom)
        energy = float(np.linalg.norm(basis @ q + basis_bar @ (xi @ poly), "fro") ** 2 / denom**2)
        return {
            "iteration": int(iteration),
            "energy": energy,
            "diff": float(diff),
            "relative_reconstruction_error": relative_reconstruction_error,
            "nfev_total": int(nfev_total),
            "nfev_max": int(nfev_max),
        }

    initial_energy = float(np.linalg.norm(basis @ q + basis_bar @ (xi @ poly), "fro") ** 2 / denom**2)
    history = [metrics(0, initial_energy, 0, 0)]
    old_energy = 0.0
    converged = False

    if progress:
        print(
            "[KDV-MAM][AM] initial "
            f"energy={history[-1]['energy']:.6e}, recon={history[-1]['relative_reconstruction_error']:.6e}"
        )

    for iteration in range(1, int(max_iter) + 1):
        # Step 1: orthogonal Procrustes update for Omega = [V, Vbar].
        stacked_coordinates = np.vstack([q, xi @ poly])
        left, _, right = np.linalg.svd(centered @ stacked_coordinates.T, full_matrices=False)
        omega = left @ right
        basis = omega[:, :r]
        basis_bar = omega[:, r:total]

        # Step 2: linear least-squares update for Xi.
        xi, poly = _xi_normal_equation(basis, basis_bar, centered, q, p, gamma)

        # Step 3: independent nonlinear least-squares updates for q_j.
        basis_bar_xi = basis_bar @ xi
        nfev_total = 0
        nfev_max = 0
        for snapshot_index in range(train_snapshots.shape[1]):
            if coordinate_initialization == "previous_time" and snapshot_index > 0:
                initial_guess = q[:, snapshot_index - 1]
            else:
                initial_guess = q[:, snapshot_index]
            result = _solve_centered_coordinates(
                centered[:, snapshot_index],
                initial_guess,
                basis,
                basis_bar_xi,
                xi,
                p,
                ls_ftol,
                ls_max_nfev=ls_max_nfev,
            )
            q[:, snapshot_index] = result.x
            nfev_total += int(result.nfev)
            nfev_max = max(nfev_max, int(result.nfev))
        poly = polynomial_embedding(q, p)

        energy = float(np.linalg.norm(basis @ q + basis_bar @ (xi @ poly), "fro") ** 2 / denom**2)
        diff = abs(energy - old_energy)
        history.append(metrics(iteration, diff, nfev_total, nfev_max))
        if progress:
            print(
                "[KDV-MAM][AM] "
                f"iter={iteration:03d}, energy={history[-1]['energy']:.6e}, "
                f"recon={history[-1]['relative_reconstruction_error']:.6e}, "
                f"diff={diff:.3e}, nfev_total={nfev_total}"
            )
        if diff < float(tol):
            converged = True
            break
        old_energy = energy

    reconstruction = reconstruct_mam_snapshots(q, basis, basis_bar, xi, u_ref, p)
    relative_reconstruction_error = float(np.linalg.norm(train_snapshots - reconstruction, "fro") / denom)
    energy_captured = float(np.linalg.norm(basis @ q + basis_bar @ (xi @ poly), "fro") ** 2 / denom**2)

    return {
        "basis": basis,
        "basis_bar": basis_bar,
        "pod_basis": pod_basis,
        "sigma": sigma,
        "u_ref": u_ref,
        "q": q,
        "poly": poly,
        "xi": xi,
        "reconstruction": reconstruction,
        "relative_reconstruction_error": relative_reconstruction_error,
        "energy_captured": energy_captured,
        "history": history,
        "converged": bool(converged),
        "iterations": int(history[-1]["iteration"]),
        "initial_mpod_reconstruction_error": float(initial["relative_reconstruction_error"]),
        "initial_mpod_energy_captured": float(initial["energy_captured"]),
    }


def history_to_arrays(history):
    """Convert MAM history dictionaries to numeric arrays for saving."""
    keys = (
        "iteration",
        "energy",
        "diff",
        "relative_reconstruction_error",
        "nfev_total",
        "nfev_max",
    )
    return {f"history_{key}": np.asarray([row[key] for row in history]) for key in keys}


def plot_basis_comparison(x, pod_basis, mam_basis, out_path, num_vectors=4):
    """Plot leading POD and MAM basis vectors for inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = np.asarray(x, dtype=np.float64)
    pod_basis = np.asarray(pod_basis, dtype=np.float64)
    mam_basis = np.asarray(mam_basis, dtype=np.float64)
    ncols = min(int(num_vectors), pod_basis.shape[1], mam_basis.shape[1])
    fig, axes = plt.subplots(ncols, 1, figsize=(8.0, 1.8 * ncols), sharex=True)
    if ncols == 1:
        axes = [axes]
    for index, ax in enumerate(axes):
        ax.plot(x, pod_basis[:, index], color="#333333", linewidth=1.8, label="POD")
        ax.plot(x, mam_basis[:, index], color="#c44e52", linewidth=1.4, linestyle="--", label="MAM")
        ax.set_ylabel(f"v{index + 1}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("x")
    fig.suptitle("KdV leading basis vectors: POD vs MAM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_model(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, model_family=np.asarray(MODEL_FAMILY), **arrays)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAM-OpInf model not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    for key in (
        "model_family",
        "snapshot_file",
        "regularizer_convention",
        "coordinate_initialization",
        "model_variant",
        "tuning_strategy",
    ):
        if key in data:
            data[key] = str(np.asarray(data[key]).item())
    for key in (
        "num_modes",
        "total_modes",
        "num_secondary",
        "degree",
        "num_features",
        "num_higher_features",
        "am_iterations",
        "unstable_index",
        "rk4_substeps",
    ):
        if key in data:
            data[key] = int(np.asarray(data[key]).item())
    for key in (
        "dt",
        "train_final_time",
        "gamma",
        "am_tol",
        "ls_ftol",
        "ridge_c",
        "ridge_a",
        "ridge_h",
        "ridge_p",
        "energy_captured",
        "relative_reconstruction_error",
        "relative_derivative_error",
        "training_rollout_error",
        "initial_mpod_reconstruction_error",
        "initial_mpod_energy_captured",
    ):
        if key in data:
            data[key] = float(np.asarray(data[key]).item())
    if "am_converged" in data:
        data["am_converged"] = bool(np.asarray(data["am_converged"]).item())
    return data
