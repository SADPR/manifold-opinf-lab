"""Utilities for standard linear-subspace OpInf on the KdV soliton benchmark."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_FAMILY = "kdv_standard_quadratic_continuous_opinf"


def load_fom_dataset(snapshot_file):
    data = np.load(snapshot_file, allow_pickle=False)
    x = np.asarray(data["x"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    snapshots = np.asarray(data["snapshots"], dtype=np.float64)
    train_mask = np.asarray(data["train_mask"], dtype=bool)
    if snapshots.shape != (x.size, times.size):
        raise ValueError(
            f"Expected snapshots shape {(x.size, times.size)}, got {snapshots.shape}."
        )
    if train_mask.shape != times.shape:
        raise ValueError(f"Expected train_mask shape {times.shape}, got {train_mask.shape}.")
    return x, times, snapshots, train_mask, data


def compute_pod_basis(train_snapshots, num_modes):
    """Compute centered POD basis using the time-averaged training mean."""
    train_snapshots = np.asarray(train_snapshots, dtype=np.float64)
    u_ref = np.mean(train_snapshots, axis=1)
    centered = train_snapshots - u_ref[:, None]
    left, sigma, _ = np.linalg.svd(centered, full_matrices=False)
    num_modes = int(num_modes)
    if num_modes < 1 or num_modes > left.shape[1]:
        raise ValueError(f"num_modes must be in [1, {left.shape[1]}], got {num_modes}.")
    basis = left[:, :num_modes]
    energy = sigma**2
    energy_captured = float(np.sum(energy[:num_modes]) / np.sum(energy))
    return basis, sigma, u_ref, energy_captured


def project_snapshots(snapshots, basis, u_ref):
    return np.asarray(basis, dtype=np.float64).T @ (
        np.asarray(snapshots, dtype=np.float64) - np.asarray(u_ref, dtype=np.float64)[:, None]
    )


def reconstruct_snapshots(q, basis, u_ref):
    return np.asarray(u_ref, dtype=np.float64)[:, None] + np.asarray(basis, dtype=np.float64) @ np.asarray(q)


def compact_quadratic_features(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    features = []
    for i in range(q.size):
        for j in range(i, q.size):
            features.append(q[i] * q[j])
    return np.asarray(features, dtype=np.float64)


def feature_vector(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return np.concatenate(([1.0], q, compact_quadratic_features(q)))


def feature_matrix(q_samples):
    """Return Theta for q_samples with shape (r, m)."""
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim != 2:
        raise ValueError("q_samples must have shape (r, m).")
    return np.vstack([feature_vector(q_samples[:, j]) for j in range(q_samples.shape[1])])


def fourth_order_time_derivative(q, dt):
    """Fourth-order centered finite-difference derivative on interior points.

    Returns derivative samples for indices 2, ..., m-3.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("q must have shape (r, m).")
    if q.shape[1] < 5:
        raise ValueError("Need at least five time samples for fourth-order finite differences.")
    dt = float(dt)
    qdot = (-q[:, 4:] + 8.0 * q[:, 3:-1] - 8.0 * q[:, 1:-3] + q[:, :-4]) / (12.0 * dt)
    return qdot


def split_feature_counts(num_modes):
    r = int(num_modes)
    n_const = 1
    n_linear = r
    n_quadratic = r * (r + 1) // 2
    return n_const, n_linear, n_quadratic


def fit_quadratic_continuous_operator(q, qdot, ridge_c=0.0, ridge_a=0.0, ridge_h=0.0):
    """Fit dq/dt = c + A q + H q_quad by regularized least squares.

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

    theta = feature_matrix(q)
    target = qdot.T
    n_const, n_linear, n_quadratic = split_feature_counts(q.shape[0])
    penalties = np.concatenate(
        [
            np.full(n_const, float(ridge_c), dtype=np.float64),
            np.full(n_linear, float(ridge_a), dtype=np.float64),
            np.full(n_quadratic, float(ridge_h), dtype=np.float64),
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
    }


def rhs(q, coeffs):
    return np.asarray(coeffs, dtype=np.float64) @ feature_vector(q)


def rollout_rk4(q0, times, coeffs, max_norm=np.inf):
    times = np.asarray(times, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q = np.empty((q0.size, times.size), dtype=np.float64)
    q[:, 0] = q0
    unstable = False
    unstable_index = -1
    for j in range(times.size - 1):
        dt = float(times[j + 1] - times[j])
        y = q[:, j]
        k1 = rhs(y, coeffs)
        k2 = rhs(y + 0.5 * dt * k1, coeffs)
        k3 = rhs(y + 0.5 * dt * k2, coeffs)
        k4 = rhs(y + dt * k3, coeffs)
        y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(y_next)) or np.linalg.norm(y_next) > float(max_norm):
            unstable = True
            unstable_index = j + 1
            q[:, j + 1:] = np.nan
            break
        q[:, j + 1] = y_next
    return q, unstable, unstable_index


def relative_state_error(reference, approximation, u_ref=None):
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    finite = np.all(np.isfinite(approximation), axis=0)
    if not np.any(finite):
        return np.inf
    ref = reference[:, finite]
    approx = approximation[:, finite]
    if u_ref is None:
        denom = np.linalg.norm(ref)
    else:
        denom = np.linalg.norm(ref - np.asarray(u_ref, dtype=np.float64)[:, None])
    return float(np.linalg.norm(ref - approx) / max(denom, 1e-15))


def relative_error_history(reference, approximation):
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    err = np.full(reference.shape[1], np.nan, dtype=np.float64)
    for j in range(reference.shape[1]):
        if np.all(np.isfinite(approximation[:, j])):
            err[j] = np.linalg.norm(reference[:, j] - approximation[:, j]) / max(
                np.linalg.norm(reference[:, j]), 1e-15
            )
    return err


def save_model(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, model_family=np.asarray(MODEL_FAMILY), **arrays)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"OpInf model not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    for key in ("model_family", "snapshot_file", "regularizer_convention"):
        if key in data:
            data[key] = str(np.asarray(data[key]).item())
    for key in ("num_modes", "num_features", "unstable_index"):
        if key in data:
            data[key] = int(np.asarray(data[key]).item())
    for key in (
        "dt",
        "train_final_time",
        "ridge_c",
        "ridge_a",
        "ridge_h",
        "energy_captured",
        "relative_derivative_error",
        "training_rollout_error",
    ):
        if key in data:
            data[key] = float(np.asarray(data[key]).item())
    return data


def plot_singular_values(sigma, num_modes, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sigma = np.asarray(sigma, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(np.arange(1, sigma.size + 1), sigma / sigma[0], "o-", markersize=3.5)
    ax.axvspan(0.5, int(num_modes) + 0.5, color="#9ecae1", alpha=0.45, label="V")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalized singular values")
    ax.set_title("KdV training snapshot singular values")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_spacetime_comparison(x, times, fom, rom, train_final_time, out_path, title, rom_label="OpInf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, sharey=True)
    vmax = max(25.0, float(np.nanmax(fom)), float(np.nanmax(rom)))
    for ax, data, label in zip(axes, (fom, rom), ("Reference", rom_label)):
        image = ax.imshow(
            data,
            extent=[times[0], times[-1], x[0], x[-1]],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=0.0,
            vmax=vmax,
        )
        ax.axvline(float(train_final_time), color="k", linestyle="--", linewidth=1.4)
        ax.set_ylabel("x-coordinate")
        ax.set_title(label)
    axes[-1].set_xlabel("time t")
    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="solution")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_comparison(x, times, fom, rom, out_path, snapshot_times=(0.2, 1.0), rom_label="OpInf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(len(snapshot_times), 1, figsize=(8.0, 3.8 * len(snapshot_times)), sharex=True)
    if len(snapshot_times) == 1:
        axes = [axes]
    for ax, target_time in zip(axes, snapshot_times):
        index = int(np.argmin(np.abs(times - float(target_time))))
        ax.plot(x, fom[:, index], color="0.25", linewidth=2.3, label="Reference")
        ax.plot(x, rom[:, index], "--", color="#4c44aa", linewidth=2.0, label=rom_label)
        ax.set_ylabel("solution")
        ax.set_title(f"t = {times[index]:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x-coordinate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_history(times, error_history, train_final_time, out_path, title="Standard OpInf error history"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.semilogy(times, np.maximum(error_history, 1e-16), linewidth=1.8)
    ax.axvline(float(train_final_time), color="k", linestyle="--", linewidth=1.4)
    ax.set_xlabel("time t")
    ax.set_ylabel("relative state error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
