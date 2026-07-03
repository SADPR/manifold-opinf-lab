"""Spectral FOM solver and plotting utilities for the KdV soliton benchmark."""

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    ALPHA,
    BETA,
    DEALIAS,
    DT,
    ETDRK4_CONTOUR_POINTS,
    FINAL_TIME,
    NUM_POINTS,
    TRAIN_FINAL_TIME,
    XL,
    XU,
    initial_condition,
    make_periodic_grid,
)


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return f"{value:.8e}" if np.isfinite(value) else str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def periodic_wavenumbers(num_points, xl=XL, xu=XU):
    """Return Fourier wavenumbers for an endpoint-excluding periodic grid."""
    length = float(xu) - float(xl)
    dx = length / int(num_points)
    return 2.0 * np.pi * np.fft.fftfreq(int(num_points), d=dx)


def _dealias_mask(k):
    """Two-thirds dealiasing mask for integer Fourier modes on [-pi, pi)."""
    n = int(k.size)
    return np.abs(k) <= (n // 3)


def make_etdrk4_coefficients(linear_operator, dt, contour_points=ETDRK4_CONTOUR_POINTS):
    """Return ETDRK4 coefficients for u_t = L u + N(u)."""
    linear_operator = np.asarray(linear_operator, dtype=np.complex128)
    dt = float(dt)
    m = int(contour_points)
    roots = np.exp(1j * np.pi * (np.arange(1, m + 1, dtype=np.float64) - 0.5) / m)
    lr = dt * linear_operator[:, None] + roots[None, :]

    e = np.exp(dt * linear_operator)
    e2 = np.exp(0.5 * dt * linear_operator)
    q = dt * np.mean((np.exp(0.5 * lr) - 1.0) / lr, axis=1)
    f1 = dt * np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1)
    f2 = dt * np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
    f3 = dt * np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1)
    return e, e2, q, f1, f2, f3


def nonlinear_hat(v_hat, k, alpha=ALPHA, dealias=DEALIAS):
    """Fourier transform of -alpha * s * s_x, using conservative form."""
    s = np.fft.ifft(v_hat).real
    nonlinear = -0.5 * float(alpha) * (1j * k) * np.fft.fft(s * s)
    if dealias:
        nonlinear = nonlinear * _dealias_mask(k)
    return nonlinear


def solve_kdv_etdrk4(
    num_points=NUM_POINTS,
    dt=DT,
    final_time=FINAL_TIME,
    alpha=ALPHA,
    beta=BETA,
    xl=XL,
    xu=XU,
    contour_points=ETDRK4_CONTOUR_POINTS,
    dealias=DEALIAS,
    save_every=1,
):
    """Solve the periodic KdV equation and return x, t, snapshots.

    The PDE is

        s_t = -alpha s s_x - beta s_xxx

    on [xl, xu) with periodic boundary conditions.
    """
    num_points = int(num_points)
    dt = float(dt)
    final_time = float(final_time)
    save_every = int(save_every)
    if save_every < 1:
        raise ValueError("save_every must be >= 1.")

    n_steps_float = final_time / dt
    n_steps = int(round(n_steps_float))
    if not np.isclose(n_steps_float, n_steps, rtol=0.0, atol=1e-12):
        raise ValueError(f"final_time/dt must be an integer, got {n_steps_float}.")

    x = make_periodic_grid(num_points, xl, xu)
    k = periodic_wavenumbers(num_points, xl, xu)
    linear_operator = 1j * float(beta) * k**3
    e, e2, q, f1, f2, f3 = make_etdrk4_coefficients(linear_operator, dt, contour_points)

    v = np.fft.fft(initial_condition(x))
    saved_steps = np.arange(0, n_steps + 1, save_every, dtype=np.int64)
    times = saved_steps.astype(np.float64) * dt
    snaps = np.empty((num_points, saved_steps.size), dtype=np.float64)
    snaps[:, 0] = np.fft.ifft(v).real

    save_index = 1
    for step in range(1, n_steps + 1):
        nv = nonlinear_hat(v, k, alpha=alpha, dealias=dealias)
        a = e2 * v + q * nv
        na = nonlinear_hat(a, k, alpha=alpha, dealias=dealias)
        b = e2 * v + q * na
        nb = nonlinear_hat(b, k, alpha=alpha, dealias=dealias)
        c = e2 * a + q * (2.0 * nb - nv)
        nc = nonlinear_hat(c, k, alpha=alpha, dealias=dealias)
        v = e * v + f1 * nv + 2.0 * f2 * (na + nb) + f3 * nc

        if step % save_every == 0:
            snaps[:, save_index] = np.fft.ifft(v).real
            save_index += 1

    return x, times, snaps


def relative_periodic_mass(snaps, xl=XL, xu=XU):
    dx = (float(xu) - float(xl)) / snaps.shape[0]
    mass = dx * np.sum(snaps, axis=0)
    return np.abs(mass - mass[0]) / max(abs(mass[0]), 1e-15)


def relative_l2_drift(snaps, xl=XL, xu=XU):
    dx = (float(xu) - float(xl)) / snaps.shape[0]
    l2 = np.sqrt(dx * np.sum(snaps * snaps, axis=0))
    return np.abs(l2 - l2[0]) / max(abs(l2[0]), 1e-15)


def save_snapshot_npz(path, x, times, snaps, train_final_time=TRAIN_FINAL_TIME, **metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_mask = times <= float(train_final_time) + 1e-14
    np.savez(
        path,
        x=np.asarray(x, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        snapshots=np.asarray(snaps, dtype=np.float64),
        train_mask=np.asarray(train_mask, dtype=np.int64),
        train_final_time=np.asarray(float(train_final_time), dtype=np.float64),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )


def plot_spacetime(x, times, snaps, out_path, train_final_time=TRAIN_FINAL_TIME):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    image = ax.imshow(
        snaps,
        extent=[times[0], times[-1], x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=0.0,
        vmax=max(25.0, float(np.max(snaps))),
    )
    ax.axvline(float(train_final_time), color="k", linestyle="--", linewidth=1.5)
    ax.set_xlabel("time t")
    ax.set_ylabel("x-coordinate")
    ax.set_title("KdV soliton FOM")
    fig.colorbar(image, ax=ax, label="solution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_snapshots(x, times, snaps, out_path, requested_times=(0.0, TRAIN_FINAL_TIME, FINAL_TIME)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for target_time in requested_times:
        index = int(np.argmin(np.abs(times - float(target_time))))
        ax.plot(x, snaps[:, index], label=f"t={times[index]:.4f}", linewidth=2.0)
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("solution")
    ax.set_title("KdV soliton snapshots")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_invariant_drift(times, snaps, out_path, xl=XL, xu=XU):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.semilogy(times, np.maximum(relative_periodic_mass(snaps, xl, xu), 1e-16), label="mass")
    ax.semilogy(times, np.maximum(relative_l2_drift(snaps, xl, xu), 1e-16), label="L2")
    ax.set_xlabel("time t")
    ax.set_ylabel("relative drift")
    ax.set_title("KdV FOM invariant drift")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_fom_report(
    report_path,
    snapshot_file,
    x,
    times,
    snaps,
    dt=DT,
    final_time=FINAL_TIME,
    train_final_time=TRAIN_FINAL_TIME,
    alpha=ALPHA,
    beta=BETA,
    dealias=DEALIAS,
):
    train_count = int(np.sum(times <= float(train_final_time) + 1e-14))
    mass_drift = relative_periodic_mass(snaps)
    l2_drift = relative_l2_drift(snaps)
    write_txt_report(
        report_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "run_fom.py")]),
            (
                "problem",
                [
                    ("equation", "s_t = -alpha*s*s_x - beta*s_xxx"),
                    ("alpha", alpha),
                    ("beta", beta),
                    ("domain", f"[{XL}, {XU})"),
                    ("periodic", True),
                    ("initial_condition", "1 + 24 sech^2(sqrt(8) x)"),
                ],
            ),
            (
                "discretization",
                [
                    ("num_points", int(x.size)),
                    ("dt", dt),
                    ("final_time", final_time),
                    ("num_saved_snapshots", int(times.size)),
                    ("train_final_time", train_final_time),
                    ("num_training_snapshots", train_count),
                    ("method", "Fourier pseudospectral ETDRK4"),
                    ("dealias", bool(dealias)),
                ],
            ),
            (
                "diagnostics",
                [
                    ("snapshot_shape", snaps.shape),
                    ("solution_min", float(np.min(snaps))),
                    ("solution_max", float(np.max(snaps))),
                    ("max_relative_mass_drift", float(np.max(mass_drift))),
                    ("max_relative_l2_drift", float(np.max(l2_drift))),
                ],
            ),
            ("outputs", [("snapshot_npz", snapshot_file), ("summary_txt", report_path)]),
        ],
    )

