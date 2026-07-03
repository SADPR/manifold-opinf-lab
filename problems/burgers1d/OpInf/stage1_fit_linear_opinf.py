#!/usr/bin/env python3
"""Stage 1: fit a discrete-time linear operator inference ROM."""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps
from opinf_utils import (
    FEATURE_MODE,
    design_matrix,
    fit_linear_discrete_operator,
    load_pod_data,
    project_snapshots,
    save_model,
)


def _default_model_path(num_modes):
    return os.path.join(SCRIPT_DIR, "models", f"linear_discrete_n{int(num_modes)}.npz")


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


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_modes=20,
    ridge=1e-8,
    feature_mode=FEATURE_MODE,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
):
    num_modes = int(num_modes)
    if model_path is None:
        model_path = _default_model_path(num_modes)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print("        STAGE 1: FIT LINEAR DISCRETE OPINF ROM")
    print("====================================================")
    print(f"[OpInf-Stage1] num_modes={num_modes}")
    print(f"[OpInf-Stage1] ridge={float(ridge):.3e}")
    print(f"[OpInf-Stage1] feature_mode={feature_mode}")

    basis, _, u_ref, metadata, basis_path, energy_captured, energy_lost = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=num_modes,
    )
    del metadata
    mu_samples = get_snapshot_params()
    if not mu_samples:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    theta_blocks = []
    y_blocks = []
    t0 = time.time()

    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[OpInf-Stage1] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
        snaps = load_or_compute_snaps(
            mu,
            GRID_X,
            U0,
            dt,
            num_steps,
            snap_folder=snap_folder,
            verbose=False,
        )
        q = project_snapshots(snaps, basis, u_ref)
        theta_blocks.append(design_matrix(q[:, :-1], mu, feature_mode=feature_mode))
        y_blocks.append(q[:, 1:].T)

    theta = np.vstack(theta_blocks)
    y_next = np.vstack(y_blocks)
    elapsed_assembly = time.time() - t0

    print(f"[OpInf-Stage1] Design matrix shape: {theta.shape}")
    print(f"[OpInf-Stage1] Target matrix shape: {y_next.shape}")
    print("[OpInf-Stage1] Solving regularized least-squares system")

    t0 = time.time()
    fit = fit_linear_discrete_operator(theta, y_next, ridge=ridge)
    elapsed_fit = time.time() - t0

    operator = fit["operator"]
    x_mean = fit["x_mean"]
    x_scale = fit["x_scale"]
    rel_one_step = fit["relative_one_step_error"]

    save_model(
        model_path=model_path,
        operator=operator,
        x_mean=x_mean,
        x_scale=x_scale,
        num_modes=num_modes,
        ridge=ridge,
        feature_mode=feature_mode,
        train_mu=mu_samples,
        train_relative_one_step_error=rel_one_step,
        dt=dt,
        num_steps=num_steps,
        pod_basis_path=basis_path,
        energy_captured=energy_captured,
        energy_lost=energy_lost,
    )

    summary_path = os.path.join(results_dir, f"linear_discrete_n{num_modes}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_linear_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", "linear_discrete_parametric"),
                    ("num_modes", num_modes),
                    ("ridge", float(ridge)),
                    ("feature_mode", feature_mode),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("num_cells", int(NUM_CELLS)),
                    ("time_scheme", TIME_SCHEME),
                    ("num_training_parameters", len(mu_samples)),
                    ("snap_folder", snap_folder),
                    ("pod_basis_path", basis_path),
                ],
            ),
            (
                "fit",
                [
                    ("design_matrix_shape", theta.shape),
                    ("target_matrix_shape", y_next.shape),
                    ("operator_shape", operator.shape),
                    ("solve_method", fit["solve_method"]),
                    ("relative_one_step_training_error", rel_one_step),
                    ("energy_captured_used_modes", energy_captured),
                    ("energy_lost_used_modes", energy_lost),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_fit_seconds", elapsed_fit),
                ],
            ),
            (
                "outputs",
                [
                    ("model_npz", model_path),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )

    print(f"[OpInf-Stage1] Relative one-step training error: {rel_one_step:.3e}")
    print(f"[OpInf-Stage1] Saved model: {model_path}")
    print(f"[OpInf-Stage1] Saved summary: {summary_path}")
    return model_path, rel_one_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a linear discrete-time OpInf ROM for 1D Burgers.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-modes", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--feature-mode", default=FEATURE_MODE)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    args = parser.parse_args()
    main(
        pod_dir=args.pod_dir,
        num_modes=args.num_modes,
        ridge=args.ridge,
        feature_mode=args.feature_mode,
        snap_folder=args.snap_folder,
        model_path=args.model_path,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
    )
