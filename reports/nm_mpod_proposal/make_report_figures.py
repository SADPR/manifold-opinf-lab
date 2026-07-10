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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)
LAB_ROOT = Path(__file__).resolve().parents[2]
BURGERS = LAB_ROOT / "problems" / "burgers1d"
KDV = LAB_ROOT / "problems" / "kdv_soliton"
KDV_OPINF = KDV / "OpInf"
if str(KDV_OPINF) not in sys.path:
    sys.path.insert(0, str(KDV_OPINF))

from gpr_nm_mpod_opinf_utils import reconstruct_gpr_nm_mpod_snapshots


def _tag(mu1, mu2):
    return f"mu1_{mu1:.2f}_mu2_{mu2:.3f}"


def _resolve(path_str):
    """Summaries were written with paths relative to problems/burgers1d/OpInf/."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (BURGERS / "OpInf" / path).resolve()


def _parse_summary(path):
    """Parse the flat '[section]\\nkey: value' .txt summaries written by the
    run_*.py evaluation scripts into a single dict (section headers dropped;
    field names are unique in practice)."""
    values = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip()
    return values


def _read_suite_errors(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(float(row["relative_error_all_finite_snapshots"]))
    return rows


BURGERS_TEST_POINTS = [(4.56, 0.019), (4.75, 0.020), (5.19, 0.026)]
BURGERS_EXTRAPOLATION_POINT = (4.00, 0.033)


def _model_errors_at_points(results_dirs, points=BURGERS_TEST_POINTS):
    """results_dirs: list of per-point result directories (one per test point,
    since these were evaluated with individual run_*.py calls rather than a
    suite script)."""
    values = []
    for results_dir, (mu1, mu2) in zip(results_dirs, points):
        matches = list(Path(results_dir).glob(f"*_summary_{_tag(mu1, mu2)}.txt"))
        if not matches:
            raise FileNotFoundError(f"No summary matching mu=({mu1},{mu2}) under {results_dir}")
        summary = _parse_summary(matches[0])
        values.append(float(summary["relative_error_all_finite_snapshots"]))
    return values


def burgers_error_comparison():
    mus = ("(4.56, 0.019)", "(4.75, 0.020)", "(5.19, 0.026)")
    models = [
        ("Standard OpInf", [BURGERS / "Results/OpInf-Standard/Continuous/r10"] * 3),
        ("MPOD induced p=2 (tuned)", [BURGERS / "Results/OpInf-MPOD-Induced/r10_q133_p2_tuned"] * 3),
        ("ANN QC-NM-MPOD", [BURGERS / "Results/OpInf-ANN-NM-MPOD/QC/r10_q133"] * 3),
        ("ANN PQ-NM-MPOD", [BURGERS / "Results/OpInf-ANN-NM-MPOD/PQ/r10_q133"] * 3),
        ("ANN AL-NM-MPOD", [BURGERS / "Results/OpInf-ANN-NM-MPOD/AL/r10_q133"] * 3),
    ]
    models = [(label, _model_errors_at_points(dirs)) for label, dirs in models]
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
    ax.set_ylim(50, 4.0e4)
    ax.set_ylabel("number of inferred RHS features")
    ax.set_title("Reduced vector-field library size (nominal feature count)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for bar, val in zip(bars, features):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.22,
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
    mu1, mu2 = 5.19, 0.026
    cases = [
        ("Standard OpInf", BURGERS / "Results/OpInf-Standard/Continuous/r10", "standard_continuous", "#4e79a7"),
        ("MPOD induced p=2 (tuned)", BURGERS / "Results/OpInf-MPOD-Induced/r10_q133_p2_tuned", "mpod_opinf", "#59a14f"),
        ("ANN PQ-NM-MPOD", BURGERS / "Results/OpInf-ANN-NM-MPOD/PQ/r10_q133", "ann_opinf", "#f28e2b"),
        ("ANN AL-NM-MPOD", BURGERS / "Results/OpInf-ANN-NM-MPOD/AL/r10_q133", "ann_opinf", "#76b7b2"),
        ("ANN QC-NM-MPOD", BURGERS / "Results/OpInf-ANN-NM-MPOD/QC/r10_q133", "ann_opinf", "#b07aa1"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for label, results_dir, prefix, color in cases:
        summary = _parse_summary(Path(results_dir) / f"{prefix}_summary_{_tag(mu1, mu2)}.txt")
        values = np.load(_resolve(summary["error_history_npy"]))
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


def burgers_profile_overlay():
    mu1, mu2 = 4.75, 0.020
    cases = [
        ("Standard OpInf", BURGERS / "Results/OpInf-Standard/Continuous/r10", "standard_continuous", "#d62728"),
        ("Induced MPOD, $p=2$", BURGERS / "Results/OpInf-MPOD-Induced/r10_q133_p2_tuned", "mpod_opinf", "#1f77b4"),
        ("ANN QC-NM-MPOD", BURGERS / "Results/OpInf-ANN-NM-MPOD/QC/r10_q133", "ann_opinf", "#2ca02c"),
    ]
    hdm = np.load(BURGERS / "Results/param_snaps/snaps_be_mu1_4.750_mu2_0.0200.npy")
    x = (np.arange(hdm.shape[0], dtype=np.float64) + 0.5) * (100.0 / hdm.shape[0])
    indices = [0, 125, 250, 375, 500]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), sharex=True, sharey=True)
    for ax, (label, results_dir, prefix, color) in zip(axes, cases):
        summary = _parse_summary(Path(results_dir) / f"{prefix}_summary_{_tag(mu1, mu2)}.txt")
        rom = np.load(_resolve(summary["rom_snapshots_npy"]))
        for idx in indices:
            ax.plot(x, hdm[:, idx], color="black", linewidth=1.0)
            ax.plot(x, rom[:, idx], color=color, linestyle="--", linewidth=1.15)
        ax.set_title(label)
        ax.set_xlabel(r"$x$")
        ax.grid(True, alpha=0.22, linewidth=0.5)
    axes[0].set_ylabel(r"$u$")
    axes[0].plot([], [], color="black", linewidth=1.0, label="HDM")
    axes[0].plot([], [], color="black", linestyle="--", linewidth=1.15, label="ROM")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "burgers_overlay_mu475.png", dpi=240)
    plt.close(fig)


def burgers_parameter_space():
    mu1_train = np.array([4.25, 4.875, 5.50], dtype=float)
    mu2_train = np.array([0.015, 0.0225, 0.030], dtype=float)
    train = np.array([(a, b) for a in mu1_train for b in mu2_train], dtype=float)
    evaluation = np.array(BURGERS_TEST_POINTS + [BURGERS_EXTRAPOLATION_POINT], dtype=float)
    labels = [r"$\mu^{(1)}$", r"$\mu^{(2)}$", r"$\mu^{(3)}$", r"$\mu^{(e)}$"]
    offsets = [(-48, -10), (18, -8), (10, 7), (10, -2)]

    fig, ax = plt.subplots(figsize=(6.7, 7.0))
    ax.scatter(
        train[:, 0], train[:, 1], marker="o", s=130, color="black", edgecolor="black",
        linewidth=0.6, label=r"Baseline $3\times3$ grid", zorder=3,
    )
    ax.scatter(
        evaluation[:, 0], evaluation[:, 1], marker="*", s=210, color="#c1272d", edgecolor="#c1272d",
        linewidth=0.6, label="Evaluation points", zorder=4,
    )
    for label, (mu1, mu2), offset in zip(labels, evaluation, offsets):
        ax.annotate(
            label, (mu1, mu2), xytext=offset, textcoords="offset points",
            fontsize=11, color="#9f1d20",
        )
    ax.set_xlim(3.75, 6.03)
    ax.set_ylim(0.0088, 0.0372)
    ax.set_xticks([4.0, 4.5, 5.0, 5.5, 6.0])
    ax.set_yticks([0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
    ax.set_xlabel(r"$\mu_1$", fontsize=13)
    ax.set_ylabel(r"$\mu_2$", fontsize=13)
    ax.set_title("Baseline training set in parameter space", fontsize=15, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(
        frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=2, fontsize=11, handletextpad=0.8, columnspacing=1.8,
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(OUT / "burgers_parameter_space.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def burgers_spacetime_topview():
    """x-t 'top view' of the extrapolatory stress-test point mu^(3), showing
    shock propagation and where each ROM departs from the HDM."""
    mu1, mu2 = BURGERS_EXTRAPOLATION_POINT
    tag = _tag(mu1, mu2)
    hdm_path = BURGERS / "Results" / "param_snaps" / f"snaps_be_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    reference = np.load(hdm_path)

    cases = [
        ("Standard OpInf, r=10", BURGERS / "Results/OpInf-Standard/Continuous/r10", "standard_continuous", "standard_continuous_snaps"),
        ("Induced MPOD, p=2 (tuned)", BURGERS / "Results/OpInf-MPOD-Induced/r10_q133_p2_tuned", "mpod_opinf", "mpod_opinf_snaps"),
        ("ANN QC-NM-MPOD", BURGERS / "Results/OpInf-ANN-NM-MPOD/QC/r10_q133", "ann_opinf", "ann_opinf_snaps"),
    ]
    panels = [("HDM reference", reference)]
    for label, results_dir, prefix, _snap_prefix in cases:
        summary = _parse_summary(Path(results_dir) / f"{prefix}_summary_{tag}.txt")
        # Result directories were reorganized after these summaries were
        # written; trust the current results_dir + stored basename, not the
        # (possibly stale) full path recorded inside the summary.
        rom = np.load(Path(results_dir) / Path(summary["rom_snapshots_npy"]).name)
        panels.append((label, rom))

    num_cells = reference.shape[0]
    x = (np.arange(num_cells, dtype=np.float64) + 0.5) * (100.0 / num_cells)
    times = 0.05 * np.arange(reference.shape[1], dtype=np.float64)
    vmin, vmax = float(np.nanmin(reference)), float(np.nanmax(reference))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4), sharey=True)
    image = None
    for ax, (title, values) in zip(axes, panels):
        image = ax.imshow(
            values, extent=[times[0], times[-1], x[0], x[-1]], origin="lower",
            aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_xlabel("time $t$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("$x$")
    fig.subplots_adjust(right=0.90, wspace=0.08)
    cax = fig.add_axes([0.92, 0.13, 0.018, 0.72])
    fig.colorbar(image, cax=cax, label="$u(x,t)$")
    fig.savefig(OUT / "burgers_extrapolation_spacetime_topview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def kdv_summary_comparison():
    labels = ["Standard", "MPOD", "GPR QC", "GPR PQ", "GPR AL"]
    full = [5.32525255e-1, 4.43847982e-1, 8.728296e-02, 8.731981e-02, 8.732928e-02]
    pred = [5.36837850e-1, 4.47967640e-1, 8.765094e-02, 8.769652e-02, 8.770766e-02]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.3, 4.2))
    ax.bar(x - 0.18, full, 0.36, label="full window", color="#4e79a7", edgecolor="black", linewidth=0.35)
    ax.bar(x + 0.18, pred, 0.36, label="prediction window", color="#f28e2b", edgecolor="black", linewidth=0.35)
    ax.set_yscale("log")
    ax.set_ylabel("relative state error")
    ax.set_title(r"KdV soliton, $r=5$")
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


def _load_kdv_comparison_snapshots():
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

    gpr_model = np.load(KDV / "OpInf/models/gpr_nm_mpod_qc_r5_q9.npz")
    gpr_rollout = np.load(KDV / "Results/OpInf/GPR-NM-MPOD/QC/r5_q9/gpr_nm_mpod_qc_opinf_r5_q9_q_rollout.npz")
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

    return x, times, train_final_time, reference, standard, mpod, gpr


def kdv_snapshot_comparison():
    x, times, _train_final_time, reference, standard, mpod, gpr = _load_kdv_comparison_snapshots()
    panels = [
        (0.2, r"$(a)\ t=0.2$"),
        (1.0, r"$(b)\ t=1$"),
    ]
    curves = [
        ("Reference", reference, "black", "-", 1.7),
        ("OpInf", standard, "#d62728", "--", 1.35),
        ("MPOD-OpInf", mpod, "#1f77b4", "--", 1.35),
        ("GPR QC-NM-MPOD", gpr, "#2ca02c", "--", 1.35),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(4.55, 6.35), sharex=True, sharey=True)
    legend_locations = ["upper left", "upper right"]
    for ax, (target_time, subtitle), legend_loc in zip(axes, panels, legend_locations):
        idx = int(np.argmin(np.abs(times - target_time)))
        for label, values, color, linestyle, linewidth in curves:
            ax.plot(x, values[:, idx], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-5.0, 30.0)
        ax.set_ylabel("solution")
        ax.set_xticks([-np.pi, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$\pi$"])
        ax.set_xlabel(r"$x$-coordinate")
        ax.grid(False)
        ax.legend(frameon=False, loc=legend_loc, fontsize=8, handlelength=1.8)
        ax.text(0.5, -0.34, subtitle, transform=ax.transAxes, ha="center", va="top")
    fig.subplots_adjust(hspace=0.56, left=0.16, right=0.96, bottom=0.10, top=0.97)
    fig.savefig(OUT / "kdv_willcox_style_snapshots.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def kdv_vertical_spacetime_comparison():
    x, times, train_final_time, reference, standard, mpod, gpr = _load_kdv_comparison_snapshots()

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
            cmap="coolwarm",
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
    burgers_parameter_space()
    burgers_profile_overlay()
    burgers_spacetime_topview()
    kdv_snapshot_comparison()
    kdv_vertical_spacetime_comparison()
