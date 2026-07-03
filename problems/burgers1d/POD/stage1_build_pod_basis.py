#!/usr/bin/env python3
"""Stage 1: build the global POD basis for the 1D Burgers PROM."""

import os
import sys
import time
from datetime import datetime

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import POD, get_snapshot_params, load_or_compute_snaps, plot_singular_value_decay


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
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def build_snapshot_matrix(mu_samples, grid_x, u0, dt, num_steps, snap_folder):
    first = np.asarray(
        load_or_compute_snaps(mu_samples[0], grid_x, u0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )
    n_dofs, n_time = first.shape
    snaps = np.zeros((n_dofs, n_time * len(mu_samples)), dtype=np.float64)
    snaps[:, :n_time] = first

    col = n_time
    for mu in mu_samples[1:]:
        s_mu = np.asarray(
            load_or_compute_snaps(mu, grid_x, u0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        if s_mu.shape != (n_dofs, n_time):
            raise RuntimeError(
                f"Snapshot shape mismatch: expected {(n_dofs, n_time)}, got {s_mu.shape} for mu={mu}."
            )
        snaps[:, col:col + n_time] = s_mu
        col += n_time

    return snaps


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    sigma_file=os.path.join(script_dir, "sigma.npy"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    decay_plot_file=os.path.join(script_dir, "stage1_pod_singular_value_decay.png"),
    metadata_file=os.path.join(script_dir, "stage1_pod_metadata.npz"),
    report_file=os.path.join(script_dir, "stage1_pod_summary.txt"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    pod_method="svd",
    pod_tol=1e-4,
    num_modes=None,
    center=True,
    random_state=0,
):
    if pod_method not in ("svd", "rsvd"):
        raise ValueError("pod_method must be 'svd' or 'rsvd'.")
    if num_modes is not None and pod_tol is not None:
        raise ValueError("Choose either num_modes or pod_tol, not both.")
    if pod_method == "rsvd" and num_modes is None:
        raise ValueError("For pod_method='rsvd', num_modes must be provided.")

    os.makedirs(os.path.dirname(basis_file), exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    print("\n====================================================")
    print("         STAGE 1: BUILD GLOBAL 1D POD BASIS")
    print("====================================================")
    print(f"[POD-STAGE1] pod_method={pod_method}, center={center}")
    print(f"[POD-STAGE1] num_cells={NUM_CELLS}")
    if num_modes is None:
        print(f"[POD-STAGE1] truncation by pod_tol={pod_tol:.1e}")
    else:
        print(f"[POD-STAGE1] truncation by num_modes={num_modes}")

    mu_samples = get_snapshot_params()
    if not mu_samples:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")
    print(f"[POD-STAGE1] Number of training parameters: {len(mu_samples)}")

    t0 = time.time()
    snaps = build_snapshot_matrix(mu_samples, GRID_X, U0, dt, num_steps, snap_folder)
    elapsed_snapshots = time.time() - t0
    print(f"[POD-STAGE1] Snapshot matrix shape: {snaps.shape}")

    t0 = time.time()
    if num_modes is None:
        basis, sigma, info, u_ref = POD(
            snaps,
            method=pod_method,
            energy_loss=pod_tol,
            random_state=random_state,
            return_truncation_info=True,
            center=center,
            return_reference=True,
        )
    else:
        basis, sigma, info, u_ref = POD(
            snaps,
            method=pod_method,
            num_modes=num_modes,
            random_state=random_state,
            return_truncation_info=True,
            center=center,
            return_reference=True,
        )
    elapsed_pod = time.time() - t0

    if u_ref is None:
        u_ref = np.zeros(snaps.shape[0], dtype=np.float64)

    np.save(basis_file, basis)
    np.save(sigma_file, sigma)
    np.save(uref_file, u_ref)
    plot_singular_value_decay(
        sigma,
        out_path=decay_plot_file,
        max_modes=min(1000, sigma.size),
        label="1D POD Stage1",
        title="1D POD residual energy decay",
    )

    np.savez(
        metadata_file,
        n_keep=np.asarray(int(info["n_keep"]), dtype=np.int64),
        n_available=np.asarray(int(info["n_available"]), dtype=np.int64),
        energy_captured=np.asarray(float(info["energy_captured"]), dtype=np.float64),
        energy_lost=np.asarray(float(info["energy_lost"]), dtype=np.float64),
        centered=np.asarray(bool(info["centered"]), dtype=np.int64),
        reference_source=np.asarray(str(info["reference_source"])),
        pod_method=np.asarray(str(pod_method)),
        pod_tol=np.asarray(np.nan if pod_tol is None else float(pod_tol), dtype=np.float64),
        num_modes_requested=np.asarray(-1 if num_modes is None else int(num_modes), dtype=np.int64),
        num_training_parameters=np.asarray(len(mu_samples), dtype=np.int64),
        state_size=np.asarray(snaps.shape[0], dtype=np.int64),
        num_training_snapshots=np.asarray(snaps.shape[1], dtype=np.int64),
        dt=np.asarray(float(dt), dtype=np.float64),
        num_steps=np.asarray(int(num_steps), dtype=np.int64),
        num_cells=np.asarray(int(NUM_CELLS), dtype=np.int64),
        time_scheme=np.asarray(str(TIME_SCHEME)),
    )

    write_txt_report(
        report_file,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "POD/stage1_build_pod_basis.py")]),
            (
                "configuration",
                [
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("num_cells", NUM_CELLS),
                    ("time_scheme", TIME_SCHEME),
                    ("pod_method", pod_method),
                    ("pod_tol", pod_tol if num_modes is None else None),
                    ("num_modes_requested", num_modes),
                    ("center", center),
                    ("random_state", random_state),
                    ("num_training_parameters", len(mu_samples)),
                ],
            ),
            (
                "snapshot_matrix",
                [
                    ("shape", snaps.shape),
                    ("state_size", snaps.shape[0]),
                    ("num_snapshots", snaps.shape[1]),
                    ("elapsed_snapshot_loading_seconds", elapsed_snapshots),
                ],
            ),
            (
                "pod_truncation",
                [
                    ("n_keep", int(info["n_keep"])),
                    ("energy_captured", float(info["energy_captured"])),
                    ("energy_lost", float(info["energy_lost"])),
                    ("centered", bool(info["centered"])),
                    ("reference_source", info["reference_source"]),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("elapsed_pod_seconds", elapsed_pod),
                ],
            ),
            (
                "outputs",
                [
                    ("basis_npy", basis_file),
                    ("sigma_npy", sigma_file),
                    ("u_ref_npy", uref_file),
                    ("decay_plot_png", decay_plot_file),
                    ("metadata_npz", metadata_file),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )
    print(f"[POD-STAGE1] Saved basis: {basis_file}")
    print(f"[POD-STAGE1] Saved summary: {report_file}")
    return basis, sigma, u_ref


if __name__ == "__main__":
    main()
