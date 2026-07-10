#!/usr/bin/env python3
"""Stage 1: fit ANN-manifold continuous-time OpInf ROM."""

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ann_manifold_opinf_utils import (
    ann_continuous_feature_matrix,
    ann_manifold_decode,
    build_ann_input_matrix,
    model_stub_from_ann_fit,
    parse_hidden_dims,
    save_ann_manifold_model,
    train_ann_secondary_map,
)
from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps
from manifold_opinf_utils import FEATURE_MODE, estimate_time_derivative, fit_continuous_operator
from opinf_utils import load_pod_data, project_snapshots
from stage1_fit_linear_opinf import write_txt_report


def _default_model_path(num_primary, num_secondary):
    return os.path.join(SCRIPT_DIR, "models", f"ann_manifold_linear_plus_ann_r{int(num_primary)}_q{int(num_secondary)}.npz")


def _energy_captured(sigma, n_keep):
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0 or int(n_keep) > sigma.size:
        return np.nan
    total = float(np.sum(sigma**2))
    return float(np.sum(sigma[: int(n_keep)] ** 2) / total) if total > 0.0 else 1.0


def _format_bool(value):
    return "true" if bool(value) else "false"


def _plot_training_history(train_history, val_history, out_path):
    train_history = np.asarray(train_history, dtype=np.float64).reshape(-1)
    val_history = np.asarray(val_history, dtype=np.float64).reshape(-1)
    if train_history.size == 0:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = np.arange(1, train_history.size + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, train_history, label="train MSE", linewidth=1.5)
    if val_history.size == train_history.size:
        ax.plot(epochs, val_history, label="validation MSE", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def main(
    pod_dir=os.path.join(PROJECT_ROOT, "POD"),
    num_primary=20,
    num_secondary=123,
    feature_mode=FEATURE_MODE,
    snap_folder=os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    model_path=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "Training"),
    dt=DT,
    num_steps=NUM_STEPS,
    ann_include_mu=True,
    hidden_dims=(32, 64, 128, 256, 256),
    validation_fraction=0.1,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-6,
    epochs=2000,
    patience=120,
    min_improve=1e-12,
    clip_grad=1.0,
    print_every=25,
    device=None,
    random_seed=42,
    dynamics_ridge=1e2,
    include_param_linear=True,
    include_quadratic=False,
    include_higher=False,
    max_degree=2,
    include_ann_dynamics=True,
    include_param_ann_dynamics=False,
):
    num_primary = int(num_primary)
    num_secondary = int(num_secondary)
    n_total = num_primary + num_secondary
    if num_primary < 1:
        raise ValueError("--num-primary must be positive.")
    if num_secondary < 1:
        raise ValueError("--num-secondary must be positive.")
    if model_path is None:
        model_path = _default_model_path(num_primary, num_secondary)
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("\n====================================================")
    print("      STAGE 1: FIT ANN-MANIFOLD OPINF ROM")
    print("====================================================")
    print(f"[ANN-OpInf] num_primary={num_primary}, num_secondary={num_secondary}")
    print(f"[ANN-OpInf] ann_include_mu={_format_bool(ann_include_mu)}")
    print(f"[ANN-OpInf] hidden_dims={tuple(int(v) for v in hidden_dims)}")
    print(
        "[ANN-OpInf] training: "
        f"epochs={int(epochs)}, patience={int(patience)}, "
        f"batch_size={int(batch_size)}, lr={float(learning_rate):.3e}, "
        f"weight_decay={float(weight_decay):.3e}"
    )
    print(
        "[ANN-OpInf] dynamics terms: "
        f"param_linear={_format_bool(include_param_linear)}, "
        f"quadratic={_format_bool(include_quadratic)}, "
        f"higher={_format_bool(include_higher)}, "
        f"ann={_format_bool(include_ann_dynamics)}, "
        f"param_ann={_format_bool(include_param_ann_dynamics)}, "
        f"ridge={float(dynamics_ridge):.3e}"
    )

    basis_total, sigma, u_ref, metadata, basis_path, _, _ = load_pod_data(
        pod_dir,
        U0.size,
        num_modes=n_total,
    )
    del metadata
    basis_primary = basis_total[:, :num_primary]
    basis_secondary = basis_total[:, num_primary:n_total]
    mu_samples = get_snapshot_params()

    q_primary_blocks = []
    q_secondary_blocks = []
    qdot_blocks = []
    x_blocks = []
    group_id_blocks = []
    t0 = time.time()

    for imu, mu in enumerate(mu_samples, start=1):
        print(f"[ANN-OpInf] Loading/projecting mu {imu}/{len(mu_samples)}: ({mu[0]:.3f}, {mu[1]:.4f})")
        snaps = load_or_compute_snaps(
            mu,
            GRID_X,
            U0,
            dt,
            num_steps,
            snap_folder=snap_folder,
            verbose=False,
        )
        q_total = project_snapshots(snaps, basis_total, u_ref)
        q_primary = q_total[:num_primary, :]
        q_secondary = q_total[num_primary:n_total, :]
        qdot = estimate_time_derivative(q_primary, dt)
        q_primary_blocks.append(q_primary)
        q_secondary_blocks.append(q_secondary)
        qdot_blocks.append(qdot.T)
        x_blocks.append(build_ann_input_matrix(q_primary, mu, include_mu=ann_include_mu))
        group_id_blocks.append(np.full(q_primary.shape[1], imu - 1, dtype=np.int64))

    q_primary_all = np.hstack(q_primary_blocks)
    q_secondary_all = np.hstack(q_secondary_blocks)
    x_all = np.vstack(x_blocks)
    y_all = q_secondary_all.T
    group_ids_all = np.concatenate(group_id_blocks)
    elapsed_assembly = time.time() - t0

    print(f"[ANN-OpInf] ANN input matrix shape: {x_all.shape}")
    print(f"[ANN-OpInf] ANN output matrix shape: {y_all.shape}")
    print(
        "[ANN-OpInf] Training ANN decoder q_secondary = N_ANN(q_primary, mu) "
        "(validation holds out whole trajectories, not scattered snapshots)"
    )
    t0 = time.time()
    ann_fit = train_ann_secondary_map(
        x_all,
        y_all,
        hidden_dims=hidden_dims,
        validation_fraction=validation_fraction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        min_improve=min_improve,
        clip_grad=clip_grad,
        random_seed=random_seed,
        device=device,
        print_every=print_every,
        group_ids=group_ids_all,
    )
    elapsed_ann_fit = time.time() - t0
    print(
        "[ANN-OpInf] Best ANN decoder: "
        f"val_mse={ann_fit['best_val_mse']:.3e}, "
        f"train_rel={ann_fit['train_relative_error']:.3e}, "
        f"val_rel={ann_fit['validation_relative_error']:.3e}, "
        f"epochs={ann_fit['epochs_trained']}"
    )

    model_stub = model_stub_from_ann_fit(
        ann_fit,
        feature_mode=feature_mode,
        include_param_linear=include_param_linear,
        include_quadratic=include_quadratic,
        include_higher=include_higher,
        max_degree=max_degree,
        ann_include_mu=ann_include_mu,
        include_ann_dynamics=include_ann_dynamics,
        include_param_ann_dynamics=include_param_ann_dynamics,
    )

    theta_blocks = []
    for mu, q_primary in zip(mu_samples, q_primary_blocks):
        theta_blocks.append(
            ann_continuous_feature_matrix(
                q_primary,
                mu,
                model_stub,
                include_ann_dynamics=include_ann_dynamics,
                include_param_ann_dynamics=include_param_ann_dynamics,
            )
        )
    theta = np.vstack(theta_blocks)
    qdot = np.vstack(qdot_blocks)
    print(f"[ANN-OpInf] Dynamics design matrix shape: {theta.shape}")
    print(f"[ANN-OpInf] Dynamics target matrix shape: {qdot.shape}")
    print("[ANN-OpInf] Fitting continuous-time reduced operators")
    t0 = time.time()
    dyn_fit = fit_continuous_operator(theta, qdot, ridge=dynamics_ridge)
    elapsed_dynamics_fit = time.time() - t0

    err_sq = 0.0
    denom_sq = 0.0
    for mu, q_primary in zip(mu_samples, q_primary_blocks):
        snaps = load_or_compute_snaps(
            mu,
            GRID_X,
            U0,
            dt,
            num_steps,
            snap_folder=snap_folder,
            verbose=False,
        )
        recon = ann_manifold_decode(
            q_primary,
            basis_primary,
            basis_secondary,
            model_stub,
            u_ref,
            mu,
        )
        err_sq += float(np.linalg.norm(snaps - recon) ** 2)
        denom_sq += float(np.linalg.norm(snaps) ** 2)
    manifold_state_rel_error = float(np.sqrt(err_sq / denom_sq)) if denom_sq > 0.0 else 0.0

    history_plot_path = os.path.join(results_dir, f"ann_manifold_continuous_r{num_primary}_q{num_secondary}_training_mse.png")
    history_plot_saved = _plot_training_history(
        ann_fit["train_history"],
        ann_fit["val_history"],
        history_plot_path,
    )

    save_ann_manifold_model(
        model_path,
        ann_state_dict=ann_fit["state_dict"],
        operator=dyn_fit["operator"],
        x_mean=dyn_fit["x_mean"],
        x_scale=dyn_fit["x_scale"],
        num_primary=num_primary,
        num_secondary=num_secondary,
        dynamics_feature_type=(
            "linear_plus_quadratic_plus_ann"
            if include_quadratic and include_ann_dynamics
            else "linear_plus_ann"
            if include_ann_dynamics
            else "linear_plus_quadratic"
            if include_quadratic
            else "linear"
        ),
        dynamics_ridge=float(dynamics_ridge),
        feature_mode=feature_mode,
        include_param_linear=bool(include_param_linear),
        include_quadratic=bool(include_quadratic),
        include_higher=bool(include_higher),
        max_degree=int(max_degree),
        include_ann_dynamics=bool(include_ann_dynamics),
        include_param_ann_dynamics=bool(include_param_ann_dynamics),
        num_features=int(dyn_fit["operator"].shape[1]),
        ann_include_mu=bool(ann_include_mu),
        ann_hidden_dims=np.asarray(tuple(int(v) for v in hidden_dims), dtype=np.int64),
        ann_x_mean=ann_fit["x_mean"],
        ann_x_std=ann_fit["x_std"],
        ann_y_mean=ann_fit["y_mean"],
        ann_y_std=ann_fit["y_std"],
        ann_train_history=ann_fit["train_history"],
        ann_val_history=ann_fit["val_history"],
        ann_validation_fraction=float(validation_fraction),
        ann_batch_size=int(batch_size),
        ann_learning_rate=float(learning_rate),
        ann_weight_decay=float(weight_decay),
        ann_epochs_requested=int(epochs),
        ann_epochs_trained=int(ann_fit["epochs_trained"]),
        ann_patience=int(patience),
        ann_min_improve=float(min_improve),
        ann_clip_grad=float(clip_grad),
        ann_random_seed=int(random_seed),
        ann_best_val_mse=float(ann_fit["best_val_mse"]),
        ann_train_relative_error=float(ann_fit["train_relative_error"]),
        ann_validation_relative_error=float(ann_fit["validation_relative_error"]),
        ann_full_relative_error=float(ann_fit["full_relative_error"]),
        relative_manifold_training_error=manifold_state_rel_error,
        relative_derivative_training_error=dyn_fit["relative_derivative_training_error"],
        dt=float(dt),
        num_steps=int(num_steps),
        pod_basis_path=basis_path,
        energy_captured_primary=_energy_captured(sigma, num_primary),
        energy_captured_total_basis=_energy_captured(sigma, n_total),
    )

    summary_path = os.path.join(results_dir, f"ann_manifold_continuous_r{num_primary}_q{num_secondary}_training_summary.txt")
    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/stage1_fit_ann_manifold_opinf.py")]),
            (
                "configuration",
                [
                    ("model_family", "ann_manifold_continuous"),
                    ("num_primary", num_primary),
                    ("num_secondary", num_secondary),
                    ("ann_include_mu", bool(ann_include_mu)),
                    ("hidden_dims", tuple(int(v) for v in hidden_dims)),
                    ("validation_fraction", float(validation_fraction)),
                    ("batch_size", int(batch_size)),
                    ("learning_rate", float(learning_rate)),
                    ("weight_decay", float(weight_decay)),
                    ("epochs", int(epochs)),
                    ("patience", int(patience)),
                    ("min_improve", float(min_improve)),
                    ("clip_grad", float(clip_grad)),
                    ("print_every", int(print_every)),
                    ("device", ann_fit["device"]),
                    ("random_seed", int(random_seed)),
                    ("dynamics_ridge", float(dynamics_ridge)),
                    ("include_param_linear", bool(include_param_linear)),
                    ("include_quadratic", bool(include_quadratic)),
                    ("include_higher", bool(include_higher)),
                    ("max_degree", int(max_degree)),
                    ("include_ann_dynamics", bool(include_ann_dynamics)),
                    ("include_param_ann_dynamics", bool(include_param_ann_dynamics)),
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
                    ("ann_input_matrix_shape", x_all.shape),
                    ("ann_output_matrix_shape", y_all.shape),
                    ("ann_train_samples", int(ann_fit["train_indices"].size)),
                    ("ann_validation_samples", int(ann_fit["validation_indices"].size)),
                    ("ann_best_val_mse", float(ann_fit["best_val_mse"])),
                    ("ann_epochs_trained", int(ann_fit["epochs_trained"])),
                    ("ann_train_relative_error", float(ann_fit["train_relative_error"])),
                    ("ann_validation_relative_error", float(ann_fit["validation_relative_error"])),
                    ("ann_full_relative_error", float(ann_fit["full_relative_error"])),
                    ("dynamics_design_matrix_shape", theta.shape),
                    ("dynamics_target_matrix_shape", qdot.shape),
                    ("operator_shape", dyn_fit["operator"].shape),
                    ("dynamics_solve_method", dyn_fit["solve_method"]),
                    ("relative_manifold_training_error", manifold_state_rel_error),
                    ("relative_derivative_training_error", dyn_fit["relative_derivative_training_error"]),
                    ("energy_captured_primary", _energy_captured(sigma, num_primary)),
                    ("energy_captured_total_basis", _energy_captured(sigma, n_total)),
                    ("elapsed_assembly_seconds", elapsed_assembly),
                    ("elapsed_ann_fit_seconds", elapsed_ann_fit),
                    ("elapsed_dynamics_fit_seconds", elapsed_dynamics_fit),
                ],
            ),
            (
                "outputs",
                [
                    ("model_npz", model_path),
                    ("training_mse_png", history_plot_path if history_plot_saved else None),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )

    print(f"[ANN-OpInf] Manifold state training error: {manifold_state_rel_error:.3e}")
    print(f"[ANN-OpInf] Derivative training error: {dyn_fit['relative_derivative_training_error']:.3e}")
    print(f"[ANN-OpInf] Saved model: {model_path}")
    if history_plot_saved:
        print(f"[ANN-OpInf] Saved training curve: {history_plot_path}")
    print(f"[ANN-OpInf] Saved summary: {summary_path}")
    return model_path, manifold_state_rel_error, dyn_fit["relative_derivative_training_error"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit ANN-manifold continuous-time OpInf ROM.")
    parser.add_argument("--pod-dir", default=os.path.join(PROJECT_ROOT, "POD"))
    parser.add_argument("--num-primary", type=int, default=20)
    parser.add_argument("--num-secondary", type=int, default=123)
    parser.add_argument("--feature-mode", default=FEATURE_MODE)
    parser.add_argument("--snap-folder", default=os.path.join(PROJECT_ROOT, "Results", "param_snaps"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf-ANN-Manifold", "Training"))
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--no-ann-mu", action="store_true")
    parser.add_argument("--hidden-dims", default="32,64,128,256,256")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-improve", type=float, default=1e-12)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dynamics-ridge", type=float, default=1e2)
    parser.add_argument("--no-param-linear", action="store_true")
    parser.add_argument("--with-quadratic", action="store_true")
    parser.add_argument("--no-quadratic", action="store_true", help="Deprecated; quadratic dynamics are off by default.")
    parser.add_argument("--with-higher", action="store_true")
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--with-ann-dynamics", action="store_true", help="Deprecated; ANN dynamics are on by default.")
    parser.add_argument("--no-ann-dynamics", action="store_true")
    parser.add_argument("--with-param-ann-dynamics", action="store_true")
    args = parser.parse_args()
    main(
        pod_dir=args.pod_dir,
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        feature_mode=args.feature_mode,
        snap_folder=args.snap_folder,
        model_path=args.model_path,
        results_dir=args.results_dir,
        dt=args.dt,
        num_steps=args.num_steps,
        ann_include_mu=not args.no_ann_mu,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_improve=args.min_improve,
        clip_grad=args.clip_grad,
        print_every=args.print_every,
        device=args.device,
        random_seed=args.random_seed,
        dynamics_ridge=args.dynamics_ridge,
        include_param_linear=not args.no_param_linear,
        include_quadratic=bool(args.with_quadratic and not args.no_quadratic),
        include_higher=bool(args.with_higher),
        max_degree=args.max_degree,
        include_ann_dynamics=bool((not args.no_ann_dynamics) or args.with_ann_dynamics),
        include_param_ann_dynamics=bool(args.with_param_ann_dynamics),
    )
