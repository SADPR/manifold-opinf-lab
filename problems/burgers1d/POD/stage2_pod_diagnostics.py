#!/usr/bin/env python3
"""Stage 2: compute projection diagnostics for the 1D POD basis."""

import os
import sys
import tempfile
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    report_file=os.path.join(script_dir, "stage2_pod_diagnostics_summary.txt"),
    plot_file=os.path.join(script_dir, "stage2_pod_diagnostics.png"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
):
    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    if os.path.exists(uref_file):
        u_ref = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(basis.shape[0], dtype=np.float64)

    mu_samples = get_snapshot_params()
    rel_errors = []
    labels = []
    for mu in mu_samples:
        snaps = np.asarray(
            load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        centered = snaps - u_ref[:, None]
        q = basis.T @ centered
        recon = u_ref[:, None] + basis @ q
        denom = np.linalg.norm(snaps)
        err = np.linalg.norm(snaps - recon) / (denom if denom > 0.0 else 1.0)
        rel_errors.append(float(err))
        labels.append(f"({mu[0]:.2f},{mu[1]:.3f})")
        print(f"[POD-STAGE2] mu={mu}: relative projection error={err:.3e}")

    rel_errors = np.asarray(rel_errors, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(rel_errors.size), rel_errors)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(rel_errors.size))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Relative projection error")
    ax.set_title("1D POD training projection diagnostics")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=200)
    plt.close(fig)

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: POD/stage2_pod_diagnostics.py\n\n")
        file.write("[configuration]\n")
        file.write(f"basis_file: {basis_file}\n")
        file.write(f"basis_shape: {basis.shape}\n")
        file.write(f"num_cells: {NUM_CELLS}\n")
        file.write(f"time_scheme: {TIME_SCHEME}\n")
        file.write(f"num_steps: {int(num_steps)}\n")
        file.write(f"num_training_parameters: {len(mu_samples)}\n\n")
        file.write("[results]\n")
        file.write(f"max_relative_projection_error: {float(np.max(rel_errors)):.8e}\n")
        file.write(f"mean_relative_projection_error: {float(np.mean(rel_errors)):.8e}\n")
        file.write(f"min_relative_projection_error: {float(np.min(rel_errors)):.8e}\n")
        for mu, err in zip(mu_samples, rel_errors):
            file.write(f"relative_error_mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}: {err:.8e}\n")
        file.write("\n[outputs]\n")
        file.write(f"plot_png: {plot_file}\n")
        file.write(f"summary_txt: {report_file}\n")

    print(f"[POD-STAGE2] Summary saved: {report_file}")
    print(f"[POD-STAGE2] Plot saved: {plot_file}")
    return rel_errors


if __name__ == "__main__":
    main()
