#!/usr/bin/env python3
"""Run the standard linear-subspace continuous-time OpInf comparison points."""

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

from run_standard_continuous_opinf import DEFAULT_MODEL_PATH, main as run_standard_continuous_opinf


TEST_POINTS = (
    (4.56, 0.019),
    (4.75, 0.020),
    (5.19, 0.026),
)


def _plot_suite_errors(rows, out_path):
    labels = []
    errors = []
    for row in rows:
        labels.append(f"({float(row['mu1']):.2f}, {float(row['mu2']):.3f})")
        errors.append(float(row["relative_error_all_finite_snapshots"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(x, errors, color="#4c78a8", edgecolor="#2f4f6f", linewidth=0.8)
    positive = [err for err in errors if np.isfinite(err) and err > 0.0]
    if positive and max(positive) / min(positive) > 20.0:
        ax.set_yscale("log")
        ax.set_ylim(max(min(positive) * 0.5, 1e-16), max(positive) * 5.0)
        label_y = [err * 1.2 for err in errors]
    else:
        ymax = max(errors) if errors else 1.0
        ax.set_ylim(0.0, ymax * 1.35 if ymax > 0.0 else 1.0)
        label_y = [err + 0.035 * ymax for err in errors]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("(mu1, mu2)")
    ax.set_ylabel("relative state error")
    ax.set_title("Standard continuous OpInf suite rollout errors")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for bar, err, y_text in zip(bars, errors, label_y):
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_text, f"{err:.2e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(
    test_points=TEST_POINTS,
    model_path=DEFAULT_MODEL_PATH,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-Standard", "Continuous"),
):
    os.makedirs(results_dir, exist_ok=True)
    rows = []

    print("\n====================================================")
    print("      1D STANDARD CONTINUOUS OPINF TEST SUITE")
    print("====================================================")

    for i, (mu1, mu2) in enumerate(test_points, start=1):
        print("\n----------------------------------------------------")
        print(f"[Standard-Cont-SUITE] Case {i}/{len(test_points)}: mu1={mu1:.3f}, mu2={mu2:.4f}")
        print("----------------------------------------------------")
        t0 = time.time()
        _, rel_err = run_standard_continuous_opinf(
            mu1=mu1,
            mu2=mu2,
            model_path=model_path,
            results_dir=results_dir,
        )
        elapsed = time.time() - t0
        rows.append(
            {
                "mu1": f"{mu1:.8e}",
                "mu2": f"{mu2:.8e}",
                "relative_error_all_finite_snapshots": f"{rel_err:.8e}",
                "elapsed_seconds": f"{elapsed:.8e}",
            }
        )

    csv_path = os.path.join(results_dir, "standard_continuous_suite_summary.csv")
    txt_path = os.path.join(results_dir, "standard_continuous_suite_summary.txt")
    plot_path = os.path.join(results_dir, "standard_continuous_suite_errors.png")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mu1", "mu2", "relative_error_all_finite_snapshots", "elapsed_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)
    _plot_suite_errors(rows, plot_path)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("[run]\n")
        file.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("script: OpInf/run_standard_continuous_opinf_suite.py\n\n")
        file.write("[configuration]\n")
        file.write(f"model_npz: {model_path}\n\n")
        file.write("[cases]\n")
        for row in rows:
            file.write(
                "mu1={mu1}, mu2={mu2}, "
                "relative_error_all_finite_snapshots={relative_error_all_finite_snapshots}, "
                "elapsed_seconds={elapsed_seconds}\n".format(**row)
            )
        file.write("\n[outputs]\n")
        file.write(f"summary_csv: {csv_path}\n")
        file.write(f"error_summary_png: {plot_path}\n")
        file.write(f"summary_txt: {txt_path}\n")
    print(f"\n[Standard-Cont-SUITE] Summary CSV saved: {csv_path}")
    print(f"[Standard-Cont-SUITE] Error plot saved: {plot_path}")
    print(f"[Standard-Cont-SUITE] Summary TXT saved: {txt_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the standard continuous OpInf comparison points.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-Standard", "Continuous"))
    args = parser.parse_args()
    main(model_path=args.model_path, results_dir=args.results_dir)
