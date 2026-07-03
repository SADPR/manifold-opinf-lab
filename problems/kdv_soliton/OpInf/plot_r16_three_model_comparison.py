#!/usr/bin/env python3
"""Plot the KdV r=16 Standard/MPOD/MAM spacetime comparison."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mam_opinf_utils import load_model as load_mam_model
from mam_opinf_utils import reconstruct_mam_snapshots
from mpod_opinf_utils import load_model as load_mpod_model
from mpod_opinf_utils import reconstruct_mpod_snapshots
from standard_opinf_utils import load_fom_dataset
from standard_opinf_utils import load_model as load_standard_model
from standard_opinf_utils import reconstruct_snapshots


def _load_q(path):
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["q_rom"], dtype=np.float64)


def main(
    snapshot_file=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"),
    standard_model_path=os.path.join(SCRIPT_DIR, "models", "standard_quadratic_opinf_r16.npz"),
    standard_q_path=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Standard", "r16", "standard_opinf_r16_q_rollout.npz"),
    mpod_model_path=os.path.join(SCRIPT_DIR, "models", "mpod_opinf_r16_p2_q9.npz"),
    mpod_q_path=os.path.join(PROJECT_ROOT, "Results", "OpInf", "MPOD", "r16_p2_q9", "mpod_opinf_r16_p2_q9_q_rollout.npz"),
    mam_model_path=os.path.join(SCRIPT_DIR, "models", "mam_opinf_r16_p2_q9.npz"),
    mam_q_path=os.path.join(PROJECT_ROOT, "Results", "OpInf", "MAM", "r16_p2_q9", "mam_opinf_r16_p2_q9_q_rollout.npz"),
    out_path=os.path.join(PROJECT_ROOT, "Results", "OpInf", "r16_three_model_spacetime.png"),
):
    x, times, snapshots, _, data = load_fom_dataset(snapshot_file)
    train_final_time = float(np.asarray(data["train_final_time"]).item())

    standard = load_standard_model(standard_model_path)
    standard_rom = reconstruct_snapshots(
        _load_q(standard_q_path),
        np.asarray(standard["basis"], dtype=np.float64),
        np.asarray(standard["u_ref"], dtype=np.float64),
    )

    mpod = load_mpod_model(mpod_model_path)
    mpod_rom = reconstruct_mpod_snapshots(
        _load_q(mpod_q_path),
        np.asarray(mpod["basis"], dtype=np.float64),
        np.asarray(mpod["basis_bar"], dtype=np.float64),
        np.asarray(mpod["xi"], dtype=np.float64),
        np.asarray(mpod["u_ref"], dtype=np.float64),
        int(mpod["degree"]),
    )

    mam = load_mam_model(mam_model_path)
    mam_rom = reconstruct_mam_snapshots(
        _load_q(mam_q_path),
        np.asarray(mam["basis"], dtype=np.float64),
        np.asarray(mam["basis_bar"], dtype=np.float64),
        np.asarray(mam["xi"], dtype=np.float64),
        np.asarray(mam["u_ref"], dtype=np.float64),
        int(mam["degree"]),
    )

    panels = [
        ("(a) Reference", snapshots),
        ("(b) Linear-subspace OpInf, r = 16", standard_rom),
        ("(c) MPOD-OpInf, r = 16, p = 2", mpod_rom),
        ("(d) MAM-OpInf, r = 16, p = 2", mam_rom),
    ]
    vmax = max(float(np.nanmax(panel)) for _, panel in panels)
    vmax = max(25.0, vmax)

    fig, axes = plt.subplots(len(panels), 1, figsize=(8.2, 9.2), sharex=True, sharey=True)
    image = None
    for ax, (title, panel) in zip(axes, panels):
        image = ax.imshow(
            panel,
            extent=[times[0], times[-1], x[0], x[-1]],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=0.0,
            vmax=vmax,
        )
        ax.axvline(train_final_time, color="k", linestyle="--", linewidth=1.25)
        ax.set_ylabel("x-coordinate")
        ax.set_title(title, fontsize=11)
    axes[-1].set_xlabel("time t")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="solution", shrink=0.94)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[KDV-r16] comparison plot saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot KdV r=16 Standard/MPOD/MAM comparison.")
    parser.add_argument("--snapshot-file", default=os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    parser.add_argument("--standard-model-path", default=os.path.join(SCRIPT_DIR, "models", "standard_quadratic_opinf_r16.npz"))
    parser.add_argument("--standard-q-path", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Standard", "r16", "standard_opinf_r16_q_rollout.npz"))
    parser.add_argument("--mpod-model-path", default=os.path.join(SCRIPT_DIR, "models", "mpod_opinf_r16_p2_q9.npz"))
    parser.add_argument("--mpod-q-path", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "MPOD", "r16_p2_q9", "mpod_opinf_r16_p2_q9_q_rollout.npz"))
    parser.add_argument("--mam-model-path", default=os.path.join(SCRIPT_DIR, "models", "mam_opinf_r16_p2_q9.npz"))
    parser.add_argument("--mam-q-path", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "MAM", "r16_p2_q9", "mam_opinf_r16_p2_q9_q_rollout.npz"))
    parser.add_argument("--out-path", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "r16_three_model_spacetime.png"))
    args = parser.parse_args()
    main(
        snapshot_file=args.snapshot_file,
        standard_model_path=args.standard_model_path,
        standard_q_path=args.standard_q_path,
        mpod_model_path=args.mpod_model_path,
        mpod_q_path=args.mpod_q_path,
        mam_model_path=args.mam_model_path,
        mam_q_path=args.mam_q_path,
        out_path=args.out_path,
    )
