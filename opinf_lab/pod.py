"""POD projection helpers shared by benchmark problems."""

import numpy as np


def project_snapshots(snaps, basis, u_ref):
    """Project full-order snapshots into centered POD coordinates."""
    snaps = np.asarray(snaps, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if snaps.ndim != 2:
        raise ValueError(f"snaps must be 2D, got shape {snaps.shape}.")
    if basis.ndim != 2 or basis.shape[0] != snaps.shape[0]:
        raise ValueError(f"Basis/snapshot mismatch: basis {basis.shape}, snaps {snaps.shape}.")
    if u_ref.size != snaps.shape[0]:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {snaps.shape[0]}.")
    return basis.T @ (snaps - u_ref[:, None])


def reconstruct_snapshots(q_snaps, basis, u_ref):
    """Reconstruct full-order snapshots from centered POD coordinates."""
    q_snaps = np.asarray(q_snaps, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    return u_ref[:, None] + basis @ q_snaps


def energy_captured(sigma, n_keep):
    """Return cumulative POD energy captured by the first ``n_keep`` values."""
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if sigma.size == 0 or int(n_keep) > sigma.size:
        return np.nan
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else 1.0
