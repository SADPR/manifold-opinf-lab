#!/usr/bin/env python3
"""Generate manuscript figures for the NM-MPOD proposal report."""

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)
LAB_ROOT = Path(__file__).resolve().parents[2]
BURGERS = LAB_ROOT / "problems" / "burgers1d"
KDV = LAB_ROOT / "problems" / "kdv_soliton"
KDV_OPINF = KDV / "OpInf"
if str(KDV_OPINF) not in sys.path:
    sys.path.insert(0, str(KDV_OPINF))

from gpr_nm_mpod_opinf_utils import reconstruct_gpr_nm_mpod_snapshots


def _read_suite_errors(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(float(row["relative_error_all_finite_snapshots"]))
    return rows


def burgers_error_comparison():
    mus = ("(4.56, 0.019)", "(4.75, 0.020)", "(5.19, 0.026)")
    models = [
        (
            "Standard OpInf",
            _read_suite_errors(BURGERS / "Results/OpInf-Standard/Continuous/r10/standard_continuous_suite_summary.csv"),
        ),
        (
            "MPOD induced p=2",
            _read_suite_errors(BURGERS / "Results/OpInf-MPOD-Induced/r10_q133_p2/mpod_opinf_suite_summary.csv"),
        ),
        (
            "ANN QC-NM-MPOD",
            _read_suite_errors(BURGERS / "Results/OpInf-ANN-NM-MPOD/FullQuadratic/r10_q133/ann_opinf_suite_summary.csv"),
        ),
        (
            "ANN PQ-NM-MPOD",
            _read_suite_errors(BURGERS / "Results/OpInf-ANN-NM-MPOD/LatentClosure/r10_q133/ann_opinf_suite_summary.csv"),
        ),
        (
            "ANN AL-NM-MPOD",
            _read_suite_errors(BURGERS / "Results/OpInf-ANN-NM-MPOD/LiftedLinear/r10_q133/ann_opinf_suite_summary.csv"),
        ),
    ]
    x = np.arange(len(mus))
    width = 0.15
    colors = ["#4e79a7", "#59a14f", "#b07aa1", "#f28e2b", "#76b7b2"]

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    for i, (label, values) in enumerate(models):
        offsets = x + (i - (len(models) - 1) / 2) * width
        vals = np.asarray(values, dtype=float)
        bars = ax.bar(offsets, vals, width, label=label, color=colors[i], edgecolor="black", linewidth=0.35)
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.8,
                    "unstable",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )

    ax.set_yscale("log")
    ax.set_ylim(4e-3, 2e-1)
    ax.set_xticks(x)
    ax.set_xticklabels(mus)
    ax.set_ylabel("relative state error")
    ax.set_xlabel(r"test parameter $(\mu_1,\mu_2)$")
    ax.set_title("Parametric Burgers test errors")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "burgers_error_comparison.png", dpi=220)
    plt.close(fig)


def feature_count_comparison():
    labels = [
        "Standard\nr=10",
        "ANN\nAL-NM",
        "ANN\nPQ-NM",
        "MPOD\np=2",
        "MPOD\np=5",
        "ANN\nQC-NM",
    ]
    features = np.array([121, 199, 254, 276, 1281, 10495], dtype=float)
    colors = ["#4e79a7", "#76b7b2", "#f28e2b", "#59a14f", "#8cd17d", "#b07aa1"]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, features, color=colors, edgecolor="black", linewidth=0.45)
    ax.set_yscale("log")
    ax.set_ylabel("number of inferred RHS features")
    ax.set_title("Reduced vector-field library size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for bar, val in zip(bars, features):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.12,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUT / "feature_count_comparison.png", dpi=220)
    plt.close(fig)


def feature_break_even_analysis():
    r = 10
    rbar = 133
    m = 5
    standard_features = 1 + m + r + m * r + r * (r + 1) // 2
    ann_features = standard_features + rbar + r * rbar + rbar * (rbar + 1) // 2
    p_values = np.arange(2, 21, dtype=int)
    higher = np.array([155, 400, 735, 1160, 1675, 2280, 2975, 3760, 4635, 5600, 6655, 7800, 9035, 10360, 11775, 13280, 14875, 16560, 18335])
    mpod_features = standard_features + higher
    decoder_features = r * (p_values - 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(p_values, mpod_features, marker="o", color="#59a14f", linewidth=2.0, label="MPOD RHS features")
    ax.axhline(ann_features, color="#b07aa1", linestyle="--", linewidth=2.0, label="ANN QC-NM-MPOD RHS")
    ax.axvline(15, color="0.25", linestyle=":", linewidth=1.6, label="feature parity near p=15")
    ax.set_yscale("log")
    ax.set_xlabel("MPOD polynomial order p")
    ax.set_ylabel("number of RHS features")
    ax.set_title(r"Burgers feature break-even, $r=10$, $\bar r=133$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax2 = ax.twinx()
    ax2.plot(p_values, decoder_features, marker="s", color="#f28e2b", linewidth=1.5, label=r"MPOD decoder features $r(p-1)$")
    ax2.axhline(rbar, color="#f28e2b", linestyle="--", linewidth=1.2, alpha=0.75)
    ax2.set_ylabel("decoder feature dimension")
    ax2.tick_params(axis="y", colors="#a45f00")
    fig.tight_layout()
    fig.savefig(OUT / "feature_break_even_analysis.png", dpi=220)
    plt.close(fig)


def burgers_error_history_comparison():
    cases = [
        (
            "Standard OpInf",
            BURGERS
            / "Results/OpInf-Standard/Continuous/r10/standard_continuous_error_history_mu1_5.19_mu2_0.026.npy",
            "#4e79a7",
        ),
        (
            "MPOD induced p=2",
            BURGERS / "Results/OpInf-MPOD-Induced/r10_q133_p2/mpod_opinf_error_history_mu1_5.19_mu2_0.026.npy",
            "#59a14f",
        ),
        (
            "ANN PQ-NM-MPOD",
            BURGERS
            / "Results/OpInf-ANN-NM-MPOD/LatentClosure/r10_q133/ann_opinf_error_history_mu1_5.19_mu2_0.026.npy",
            "#f28e2b",
        ),
        (
            "ANN AL-NM-MPOD",
            BURGERS
            / "Results/OpInf-ANN-NM-MPOD/LiftedLinear/r10_q133/ann_opinf_error_history_mu1_5.19_mu2_0.026.npy",
            "#76b7b2",
        ),
        (
            "ANN QC-NM-MPOD",
            BURGERS
            / "Results/OpInf-ANN-NM-MPOD/FullQuadratic/r10_q133/ann_opinf_error_history_mu1_5.19_mu2_0.026.npy",
            "#b07aa1",
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for label, path, color in cases:
        values = np.load(path)
        times = 0.05 * np.arange(values.size, dtype=np.float64)
        ax.plot(times, values, label=label, color=color, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("relative state error")
    ax.set_title(r"Burgers error history, $\mu=(5.19,0.026)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "burgers_error_history_comparison_mu519.png", dpi=220)
    plt.close(fig)


def copy_burgers_overlays():
    copies = {
        "burgers_standard_mu475.png": BURGERS
        / "Results/OpInf-Standard/Continuous/r10/standard_continuous_mu1_4.75_mu2_0.020.png",
        "burgers_mpod_p2_mu475.png": BURGERS
        / "Results/OpInf-MPOD-Induced/r10_q133_p2/mpod_opinf_mu1_4.75_mu2_0.020.png",
        "burgers_fullquad_mu475.png": BURGERS
        / "Results/OpInf-ANN-NM-MPOD/FullQuadratic/r10_q133/ann_opinf_mu1_4.75_mu2_0.020.png",
    }
    for name, src in copies.items():
        shutil.copyfile(src, OUT / name)


def burgers_parameter_space():
    mu1_train = np.array([4.25, 4.875, 5.50], dtype=float)
    mu2_train = np.array([0.015, 0.0225, 0.030], dtype=float)
    train = np.array([(a, b) for a in mu1_train for b in mu2_train], dtype=float)
    test = np.array([(4.56, 0.019), (4.75, 0.020), (5.19, 0.026)], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ax.scatter(
        train[:, 0],
        train[:, 1],
        marker="s",
        s=72,
        color="#4e79a7",
        edgecolor="black",
        linewidth=0.6,
        label="training FOM trajectories",
        zorder=3,
    )
    ax.scatter(
        test[:, 0],
        test[:, 1],
        marker="*",
        s=170,
        color="#e15759",
        edgecolor="black",
        linewidth=0.6,
        label="prediction/test trajectories",
        zorder=4,
    )
    for idx, (mu1, mu2) in enumerate(test, start=1):
        ax.annotate(
            f"T{idx}",
            (mu1, mu2),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
    ax.set_xlim(4.15, 5.60)
    ax.set_ylim(0.0135, 0.0315)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Parametric Burgers training and prediction points")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(OUT / "burgers_parameter_space.png", dpi=240)
    plt.close(fig)


def kdv_summary_comparison():
    labels = ["Standard", "MPOD", "GPR QC", "GPR PQ", "GPR AL"]
    full = [5.32525255e-1, 4.43847982e-1, 8.73141200e-2, 8.73235157e-2, 8.73292771e-2]
    pred = [5.36837850e-1, 4.47967640e-1, 8.76892300e-2, 8.77005406e-2, 8.77076641e-2]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.3, 4.2))
    ax.bar(x - 0.18, full, 0.36, label="full window", color="#4e79a7", edgecolor="black", linewidth=0.35)
    ax.bar(x + 0.18, pred, 0.36, label="prediction window", color="#f28e2b", edgecolor="black", linewidth=0.35)
    ax.set_yscale("log")
    ax.set_ylabel("relative state error")
    ax.set_title("KdV soliton r=5 comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "kdv_error_summary.png", dpi=220)
    plt.close(fig)


def _mpod_polynomial_embedding(q_samples, degree):
    q_samples = np.asarray(q_samples, dtype=np.float64)
    blocks = [q_samples ** power for power in range(2, int(degree) + 1)]
    return np.vstack(blocks)


def kdv_vertical_spacetime_comparison():
    fom = np.load(KDV / "Results/FOM/kdv_soliton_fom_snapshots.npz")
    x = fom["x"]
    times = fom["times"]
    reference = fom["snapshots"]
    train_final_time = float(fom["train_final_time"])

    standard_model = np.load(KDV / "OpInf/models/standard_quadratic_opinf_r5.npz")
    standard_rollout = np.load(KDV / "Results/OpInf/Standard/r5/standard_opinf_r5_q_rollout.npz")
    standard = standard_model["u_ref"][:, None] + standard_model["basis"] @ standard_rollout["q_rom"]

    mpod_model = np.load(KDV / "OpInf/models/mpod_opinf_r5_p2_q9.npz")
    mpod_rollout = np.load(KDV / "Results/OpInf/MPOD/r5_p2_q9/mpod_opinf_r5_p2_q9_q_rollout.npz")
    g_mpod = _mpod_polynomial_embedding(mpod_rollout["q_rom"], int(mpod_model["degree"]))
    mpod = (
        mpod_model["u_ref"][:, None]
        + mpod_model["basis"] @ mpod_rollout["q_rom"]
        + mpod_model["basis_bar"] @ (mpod_model["xi"] @ g_mpod)
    )

    gpr_model = np.load(KDV / "OpInf/models/gpr_nm_mpod_fullquadratic_r5_q9.npz")
    gpr_rollout = np.load(
        KDV
        / "Results/OpInf/GPR-NM-MPOD/FullQuadratic/r5_q9/gpr_nm_mpod_full_quadratic_opinf_r5_q9_q_rollout.npz"
    )
    gpr = reconstruct_gpr_nm_mpod_snapshots(
        gpr_rollout["q_rom"],
        gpr_model["basis"],
        gpr_model["basis_bar"],
        gpr_model["alpha"],
        gpr_model["u_ref"],
        gpr_model["centers"],
        str(gpr_model["kernel"]),
        float(gpr_model["epsilon"]),
        gpr_model["q_mean"],
        gpr_model["q_scale"],
        signal_variance=float(gpr_model["signal_variance"]),
    )

    panels = [
        ("Reference", reference),
        ("Linear-subspace OpInf", standard),
        ("MPOD-OpInf", mpod),
        ("GPR QC-NM-MPOD", gpr),
    ]
    vmin = float(np.nanmin(reference))
    vmax = float(np.nanmax(reference))
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.8), sharex=True, sharey=True)
    image = None
    for ax, (title, values) in zip(axes, panels):
        image = ax.imshow(
            values,
            extent=[times[0], times[-1], x[0], x[-1]],
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.axvline(train_final_time, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylabel(r"$x$")
        ax.set_title(title, loc="left", fontsize=10, pad=3)
    axes[-1].set_xlabel(r"time $t$")
    fig.subplots_adjust(right=0.86, hspace=0.34)
    cax = fig.add_axes([0.88, 0.14, 0.025, 0.72])
    fig.colorbar(image, cax=cax, label="solution")
    fig.savefig(OUT / "kdv_reference_linear_mpod_gpr_vertical.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    burgers_error_comparison()
    feature_count_comparison()
    feature_break_even_analysis()
    burgers_error_history_comparison()
    copy_burgers_overlays()
    burgers_parameter_space()
    kdv_summary_comparison()
    kdv_vertical_spacetime_comparison()
