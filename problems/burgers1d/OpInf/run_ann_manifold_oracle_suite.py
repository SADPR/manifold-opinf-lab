#!/usr/bin/env python3
"""Run ANN-manifold OpInf oracle diagnostics on the standard comparison points."""

import argparse
import csv
import os
import sys
import tempfile
import time
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_ann_manifold_opinf import DEFAULT_MODEL_PATH
from run_ann_manifold_oracle import main as run_ann_oracle


TEST_POINTS = (
    (4.56, 0.019),
    (4.75, 0.020),
    (5.19, 0.026),
)


SUMMARY_FIELDS = [
    "mu1",
    "mu2",
    "pod_floor_state_error",
    "ann_decoder_on_true_q_state_error",
    "ann_qbar_on_true_q_relative_error",
    "opinf_rollout_state_error",
    "true_qbar_rollout_state_error",
    "opinf_primary_q_relative_error",
    "true_qbar_primary_q_relative_error",
    "rhs_ann_on_true_q_relative_error",
    "rhs_true_qbar_on_true_q_relative_error",
    "stable_steps_opinf",
    "stable_steps_true_qbar",
    "elapsed_seconds",
]


def _plot_metric_bars(rows, out_path):
    labels = [f"({float(row['mu1']):.2f}, {float(row['mu2']):.3f})" for row in rows]
    metrics = [
        ("POD floor", "pod_floor_state_error"),
        ("ANN decoder true q", "ann_decoder_on_true_q_state_error"),
        ("ANN OpInf rollout", "opinf_rollout_state_error"),
        ("true qbar rollout", "true_qbar_rollout_state_error"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (name, key) in enumerate(metrics):
        vals = [float(row[key]) for row in rows]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel("(mu1, mu2)")
    ax.set_ylabel("relative state error")
    ax.set_title("ANN-Manifold OpInf oracle diagnostics")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(
    test_points=TEST_POINTS,
    model_path=DEFAULT_MODEL_PATH,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "Oracle"),
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    rows = []

    print("\n====================================================")
    print("      1D ANN-MANIFOLD OPINF ORACLE TEST SUITE")
    print("====================================================")

    for i, (mu1, mu2) in enumerate(test_points, start=1):
        print("\n----------------------------------------------------")
        print(f"[ANN-Oracle-SUITE] Case {i}/{len(test_points)}: mu1={mu1:.3f}, mu2={mu2:.4f}")
        print("----------------------------------------------------")
        t0 = time.time()
        metrics = run_ann_oracle(
            mu1=mu1,
            mu2=mu2,
            model_path=model_path,
            results_dir=results_dir,
            device=device,
        )
        elapsed = time.time() - t0
        row = {key: metrics.get(key, np.nan) for key in SUMMARY_FIELDS if key != "elapsed_seconds"}
        row["elapsed_seconds"] = float(elapsed)
        rows.append(row)

    csv_path = os.path.join(results_dir, "ann_oracle_suite_summary.csv")
    txt_path = os.path.join(results_dir, "ann_oracle_suite_summary.txt")
    plot_path = os.path.join(results_dir, "ann_oracle_suite_errors.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _plot_metric_bars(rows, plot_path)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: OpInf/run_ann_manifold_oracle_suite.py\n\n")
        file.write("[configuration]\n")
        file.write(f"model_npz: {model_path}\n\n")
        file.write("[cases]\n")
        for row in rows:
            file.write(
                "mu1={mu1:.8e}, mu2={mu2:.8e}, "
                "pod_floor={pod_floor_state_error:.8e}, "
                "ann_decoder={ann_decoder_on_true_q_state_error:.8e}, "
                "opinf={opinf_rollout_state_error:.8e}, "
                "true_qbar={true_qbar_rollout_state_error:.8e}, "
                "rhs_ann={rhs_ann_on_true_q_relative_error:.8e}, "
                "rhs_true_qbar={rhs_true_qbar_on_true_q_relative_error:.8e}\n".format(**row)
            )
        file.write("\n[outputs]\n")
        file.write(f"summary_csv: {csv_path}\n")
        file.write(f"summary_plot_png: {plot_path}\n")
        file.write(f"summary_txt: {txt_path}\n")

    print(f"\n[ANN-Oracle-SUITE] Summary CSV saved: {csv_path}")
    print(f"[ANN-Oracle-SUITE] Summary plot saved: {plot_path}")
    print(f"[ANN-Oracle-SUITE] Summary TXT saved: {txt_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standard ANN-manifold OpInf oracle diagnostics.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "Oracle"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(model_path=args.model_path, results_dir=args.results_dir, device=args.device)
