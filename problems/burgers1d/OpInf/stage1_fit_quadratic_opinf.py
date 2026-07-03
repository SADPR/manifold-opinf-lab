#!/usr/bin/env python3
"""Stage 1: fit a discrete-time quadratic operator inference ROM."""

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
    QUADRATIC_MODEL_FAMILY,
    design_matrix_quadratic,
    fit_linear_discrete_operator,
    load_pod_data,
    project_snapshots,
    save_model,
)
from stage1_fit_linear_opinf import write_txt_report


def _default_model_path(num_modes):
    return os.path.join(SCRIPT_DIR, "models", f"quadratic_discrete_n{int(num_modes)}.npz")


def _quadratic_feature_count(num_modes, num_eta=5, quadratic_param_mode="constant_H"):
    n = int(num_modes)
    n_quad = n * (n + 1) // 2
    count = 1 + int(num_eta) + n + n_quad
    if quadratic_param_mode in ("constant_H", "parametric_H"):
        count += int(num_eta) * n
    elif quadratic_param_mode != "constant_AH":
        raise ValueError("quadratic_param_mode must be 'constant_AH', 'constant_H', or 'parametric_H'.")
    if quadratic_param_mode == "parametric_H":
        count += int(num_eta) * n_quad
    return count


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_modes=10,
    ridge=1e-2,
    feature_mode=FEATURE_MODE,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-Quadratic", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    max_features=10000,
    quadratic_param_mode="constant_H",
):
    num_modes = int(num_modes)
    n_features = _quadratic_feature_count(num_modes, quadratic_param_mode=quadratic_param_mode)
    if n_features > int(max_features):
        raise ValueError(
            f"Quadratic feature count {n_features} exceeds --max-features={int(max_features)}. "
            "Use fewer modes or raise --max-features intentionally."
        )

    if model_path is None:
        model_path = _default_model_path(num_modes)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print("      STAGE 1: FIT QUADRATIC DISCRETE OPINF ROM")
    print("====================================================")
    print(f"[OpInf-Quad-Stage1] num_modes={num_modes}")
    print(f"[OpInf-Quad-Stage1] feature_count={n_features}")
    print(f"[OpInf-Quad-Stage1] ridge={float(ridge):.3e}")
    print(f"[OpInf-Quad-Stage1] feature_mode={feature_mode}")
    print(f"[OpInf-Quad-Stage1] quadratic_param_mode={quadratic_param_mode}")

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
        print(f"[OpInf-Quad-Stage1] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
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
        theta_blocks.append(
            design_matrix_quadratic(
                q[:, :-1],
                mu,
                feature_mode=feature_mode,
                include_param_linear=(quadratic_param_mode in ("constant_H", "parametric_H")),
                include_param_quad=(quadratic_param_mode == "parametric_H"),
            )
        )
        y_blocks.append(q[:, 1:].T)

    theta = np.vstack(theta_blocks)
    y_next = np.vstack(y_blocks)
    elapsed_assembly = time.time() - t0

    print(f"[OpInf-Quad-Stage1] Design matrix shape: {theta.shape}")
    print(f"[OpInf-Quad-Stage1] Target matrix shape: {y_next.shape}")
    print("[OpInf-Quad-Stage1] Solving regularized least-squares system")

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
        model_family=QUADRATIC_MODEL_FAMILY,
        quadratic_param_mode=quadratic_param_mode,
    )

    summary_path = os.path.join(results_dir, f"quadratic_discrete_n{num_modes}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_quadratic_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", QUADRATIC_MODEL_FAMILY),
                    ("num_modes", num_modes),
                    ("ridge", float(ridge)),
                    ("feature_mode", feature_mode),
                    ("quadratic_param_mode", quadratic_param_mode),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("num_cells", int(NUM_CELLS)),
                    ("time_scheme", TIME_SCHEME),
                    ("num_training_parameters", len(mu_samples)),
                    ("snap_folder", snap_folder),
                    ("pod_basis_path", basis_path),
                    ("max_features", int(max_features)),
                ],
            ),
            (
                "fit",
                [
                    ("design_matrix_shape", theta.shape),
                    ("target_matrix_shape", y_next.shape),
                    ("operator_shape", operator.shape),
                    ("feature_count", n_features),
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

    print(f"[OpInf-Quad-Stage1] Relative one-step training error: {rel_one_step:.3e}")
    print(f"[OpInf-Quad-Stage1] Saved model: {model_path}")
    print(f"[OpInf-Quad-Stage1] Saved summary: {summary_path}")
    return model_path, rel_one_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a quadratic discrete-time OpInf ROM for 1D Burgers.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-modes", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--feature-mode", default=FEATURE_MODE)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-Quadratic", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument(
        "--quadratic-param-mode",
        choices=("constant_AH", "constant_H", "parametric_H"),
        default="constant_H",
        help=(
            "constant_AH fits constant linear/quadratic operators with parameter affine terms; "
            "constant_H also fits parameter-linear products; parametric_H also fits parameter-quadratic products."
        ),
    )
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
        max_features=args.max_features,
        quadratic_param_mode=args.quadratic_param_mode,
    )
