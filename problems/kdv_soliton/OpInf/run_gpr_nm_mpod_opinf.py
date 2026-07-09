#!/usr/bin/env python3
"""Run GPR Nonlinear-Map MPOD-OpInf for the KdV soliton."""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from gpr_nm_mpod_opinf_utils import (
    gp_posterior_variance,
    load_fom_dataset,
    load_model,
    plot_error_history,
    plot_snapshot_comparison,
    plot_spacetime_comparison,
    project_snapshots,
    reconstruct_gpr_nm_mpod_snapshots,
    relative_error_history,
    relative_state_error,
    rollout_rk4,
)
from kdv.core import write_txt_report


def plot_gpr_uncertainty(times, posterior_std, secondary_p95, state_p95_max, train_final_time, output_path, label="GPR-NM-MPOD"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times, posterior_std, color="tab:blue", linewidth=1.8, label="posterior std")
    axes[0].plot(times, secondary_p95, color="tab:orange", linewidth=1.6, label="95% half-width")
    axes[0].axvline(train_final_time, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("secondary GP")
    axes[0].set_title(f"{label} posterior uncertainty")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(times, state_p95_max, color="tab:red", linewidth=1.8, label="max_x state 95% half-width")
    axes[1].axvline(train_final_time, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gpr_uncertainty_bands(x, times, snapshots, rom, state_p95, snapshot_times, output_path, label="GPR-NM-MPOD"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(snapshot_times), figsize=(6 * len(snapshot_times), 10), squeeze=False)
    for col, target_time in enumerate(snapshot_times):
        ax_state = axes[0, col]
        ax_residual = axes[1, col]
        ax_band = axes[2, col]
        idx = int(np.argmin(np.abs(times - float(target_time))))
        band = state_p95[:, idx]
        residual = snapshots[:, idx] - rom[:, idx]

        ax_state.fill_between(
            x,
            rom[:, idx] - band,
            rom[:, idx] + band,
            color="tab:red",
            alpha=0.25,
            linewidth=0.0,
            label="95% GP band",
        )
        ax_state.plot(x, rom[:, idx] - band, color="tab:red", alpha=0.45, linewidth=0.8)
        ax_state.plot(x, rom[:, idx] + band, color="tab:red", alpha=0.45, linewidth=0.8)
        ax_state.plot(x, snapshots[:, idx], color="0.15", linewidth=2.0, label="Reference")
        ax_state.plot(x, rom[:, idx], color="tab:red", linestyle="--", linewidth=2.0, label=label)
        ax_state.set_title(f"state, t = {times[idx]:.3f}")
        ax_state.set_ylabel("u")
        ax_state.grid(True, alpha=0.3)
        ax_state.legend(loc="best")

        ax_residual.fill_between(
            x,
            -band,
            band,
            color="tab:red",
            alpha=0.25,
            linewidth=0.0,
            label="95% GP band",
        )
        ax_residual.plot(x, -band, color="tab:red", alpha=0.45, linewidth=0.8)
        ax_residual.plot(x, band, color="tab:red", alpha=0.45, linewidth=0.8)
        ax_residual.axhline(0.0, color="0.25", linewidth=1.0)
        ax_residual.plot(x, residual, color="tab:blue", linewidth=1.6, label=f"Reference - {label}")
        limit = 1.10 * max(float(np.nanmax(np.abs(residual))), float(np.nanmax(band)), 1e-12)
        ax_residual.set_ylim(-limit, limit)
        ax_residual.set_title("residual scale")
        ax_residual.set_xlabel("x")
        ax_residual.set_ylabel("error in u")
        ax_residual.grid(True, alpha=0.3)
        ax_residual.legend(loc="best")

        ax_band.fill_between(x, -band, band, color="tab:red", alpha=0.30, linewidth=0.0, label="95% GP band")
        ax_band.plot(x, -band, color="tab:red", alpha=0.70, linewidth=1.0)
        ax_band.plot(x, band, color="tab:red", alpha=0.70, linewidth=1.0)
        ax_band.axhline(0.0, color="0.25", linewidth=1.0)
        band_limit = 1.20 * max(float(np.nanmax(band)), 1e-12)
        ax_band.set_ylim(-band_limit, band_limit)
        ax_band.set_title("GP band scale")
        ax_band.set_xlabel("x")
        ax_band.set_ylabel("u")
        ax_band.grid(True, alpha=0.3)
        ax_band.legend(loc="best")
    fig.suptitle(f"{label} posterior 95% state bands", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(
    model_path=os.path.join(SCRIPT_DIR, "models", "gpr_nm_mpod_latentclosure_r5_q9.npz"),
    snapshot_file=None,
    results_dir=os.path.join(PROJECT_ROOT, "Results", "OpInf", "GPR-NM-MPOD", "LatentClosure", "r5_q9"),
    max_norm=1e6,
    compute_uncertainty=True,
):
    model = load_model(model_path)
    if snapshot_file is None:
        snapshot_file = model.get("snapshot_file", os.path.join(PROJECT_ROOT, "Results", "FOM", "kdv_soliton_fom_snapshots.npz"))
    os.makedirs(results_dir, exist_ok=True)

    x, times, snapshots, train_mask, _ = load_fom_dataset(snapshot_file)
    basis = np.asarray(model["basis"], dtype=np.float64)
    basis_bar = np.asarray(model["basis_bar"], dtype=np.float64)
    u_ref = np.asarray(model["u_ref"], dtype=np.float64)
    alpha = np.asarray(model["alpha"], dtype=np.float64)
    centers = np.asarray(model["centers"], dtype=np.float64)
    q_mean = np.asarray(model["q_mean"], dtype=np.float64)
    q_scale = np.asarray(model["q_scale"], dtype=np.float64)
    coeffs = np.asarray(model["coeffs"], dtype=np.float64)
    kernel = str(model["kernel"])
    epsilon = float(model["epsilon"])
    noise = float(model.get("noise", 0.0))
    jitter = float(model.get("jitter", 1e-12))
    signal_variance = float(model.get("signal_variance", 1.0))
    num_modes = int(model["num_modes"])
    num_secondary = int(model["num_secondary"])
    train_final_time = float(model["train_final_time"])
    rk4_substeps = int(model.get("rk4_substeps", 1))
    include_quadratic = bool(model.get("include_quadratic", True))
    include_full_quadratic = bool(model.get("include_full_quadratic", False))
    operator_mode = str(model.get("operator_mode", "latent_closure" if include_quadratic else "lifted_linear"))
    if include_full_quadratic or operator_mode == "full_quadratic":
        label = "GPR-NM-MPOD-OpInf (full quadratic)"
        file_prefix = "gpr_nm_mpod_full_quadratic_opinf"
        dynamics = "dq/dt = c + A q + Hqq q_quad + B z_GPR(q) + Hqz (q kron z_GPR(q)) + Hzz z_quad"
    elif include_quadratic:
        label = "GPR-NM-MPOD-OpInf (latent closure)"
        file_prefix = "gpr_nm_mpod_latent_closure_opinf"
        dynamics = "dq/dt = c + A q + H q_quad + B z_GPR(q)"
    else:
        label = "GPR-NM-MPOD-OpInf (lifted linear)"
        file_prefix = "gpr_nm_mpod_lifted_linear_opinf"
        dynamics = "dq/dt = c + A q + B z_GPR(q)"

    q_fom = project_snapshots(snapshots, basis, u_ref)
    q_rom, unstable, unstable_index = rollout_rk4(
        q_fom[:, 0],
        times,
        coeffs,
        centers,
        alpha,
        kernel,
        epsilon,
        q_mean,
        q_scale,
        signal_variance=signal_variance,
        max_norm=max_norm,
        substeps=rk4_substeps,
        include_quadratic=include_quadratic,
        include_full_quadratic=include_full_quadratic,
    )
    rom = reconstruct_gpr_nm_mpod_snapshots(
        q_rom,
        basis,
        basis_bar,
        alpha,
        u_ref,
        centers,
        kernel,
        epsilon,
        q_mean,
        q_scale,
        signal_variance=signal_variance,
    )

    prediction_mask = times >= train_final_time - 1e-14
    train_error = relative_state_error(snapshots[:, train_mask], rom[:, train_mask], u_ref=u_ref)
    prediction_error = relative_state_error(snapshots[:, prediction_mask], rom[:, prediction_mask], u_ref=u_ref)
    full_error = relative_state_error(snapshots, rom, u_ref=u_ref)
    error_history = relative_error_history(snapshots, rom)

    posterior_variance = np.full(times.shape, np.nan, dtype=np.float64)
    posterior_std = np.full(times.shape, np.nan, dtype=np.float64)
    secondary_p95 = np.full(times.shape, np.nan, dtype=np.float64)
    state_p95_max = np.full(times.shape, np.nan, dtype=np.float64)
    state_p95_rms = np.full(times.shape, np.nan, dtype=np.float64)
    posterior_std_max = np.nan
    posterior_std_time_p95 = np.nan
    train_posterior_std_max = np.nan
    prediction_posterior_std_max = np.nan
    state_p95_max_over_time = np.nan
    state_p95_time_p95 = np.nan
    state_p95 = None
    if compute_uncertainty:
        posterior_variance = gp_posterior_variance(
            q_rom,
            centers,
            kernel,
            epsilon,
            q_mean,
            q_scale,
            noise=noise,
            jitter=jitter,
            signal_variance=signal_variance,
        )
        posterior_std = np.sqrt(posterior_variance)
        secondary_p95 = 1.96 * posterior_std
        basis_bar_row_norm = np.linalg.norm(basis_bar, axis=1)
        state_p95 = basis_bar_row_norm[:, None] * secondary_p95[None, :]
        state_p95_max = np.nanmax(state_p95, axis=0)
        state_p95_rms = np.sqrt(np.nanmean(state_p95**2, axis=0))

        finite_uncertainty = np.isfinite(posterior_std)
        train_uncertainty_mask = train_mask & finite_uncertainty
        prediction_uncertainty_mask = prediction_mask & finite_uncertainty
        posterior_std_max = float(np.nanmax(posterior_std))
        posterior_std_time_p95 = float(np.nanpercentile(posterior_std, 95.0))
        train_posterior_std_max = float(np.nanmax(posterior_std[train_uncertainty_mask]))
        prediction_posterior_std_max = float(np.nanmax(posterior_std[prediction_uncertainty_mask]))
        state_p95_max_over_time = float(np.nanmax(state_p95_max))
        state_p95_time_p95 = float(np.nanpercentile(state_p95_max, 95.0))

    tag = f"r{num_modes}_q{num_secondary}"
    spacetime_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_spacetime.png")
    snapshot_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_snapshots.png")
    error_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_error_history.png")
    uncertainty_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_uncertainty.png")
    uncertainty_bands_plot = os.path.join(results_dir, f"{file_prefix}_{tag}_uncertainty_bands.png")
    q_path = os.path.join(results_dir, f"{file_prefix}_{tag}_q_rollout.npz")
    summary_path = os.path.join(results_dir, f"{file_prefix}_{tag}_summary.txt")

    plot_spacetime_comparison(
        x,
        times,
        snapshots,
        rom,
        train_final_time,
        spacetime_plot,
        title=rf"KdV {label}, r={num_modes}, $\bar r$={num_secondary}",
        rom_label=label,
    )
    plot_snapshot_comparison(
        x,
        times,
        snapshots,
        rom,
        snapshot_plot,
        snapshot_times=(train_final_time, times[-1]),
        rom_label=label,
    )
    plot_error_history(
        times,
        error_history,
        train_final_time,
        error_plot,
        title=rf"{label} error history, r={num_modes}, $\bar r$={num_secondary}",
    )
    if compute_uncertainty:
        plot_gpr_uncertainty(times, posterior_std, secondary_p95, state_p95_max, train_final_time, uncertainty_plot, label=label)
        plot_gpr_uncertainty_bands(
            x,
            times,
            snapshots,
            rom,
            state_p95,
            (train_final_time, times[-1]),
            uncertainty_bands_plot,
            label=label,
        )
    np.savez(
        q_path,
        times=times,
        q_fom=q_fom,
        q_rom=q_rom,
        error_history=error_history,
        posterior_variance=posterior_variance,
        posterior_std=posterior_std,
        secondary_p95=secondary_p95,
        state_p95_max=state_p95_max,
        state_p95_rms=state_p95_rms,
        unstable=np.asarray(int(bool(unstable)), dtype=np.int64),
        unstable_index=np.asarray(int(unstable_index), dtype=np.int64),
    )

    write_txt_report(
        summary_path,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "OpInf/run_gpr_nm_mpod_opinf.py")]),
            (
                "model",
                [
                    ("model_path", model_path),
                    ("operator_mode", operator_mode),
                    ("snapshot_file", snapshot_file),
                    ("num_modes_r", num_modes),
                    ("num_secondary_rbar", num_secondary),
                    ("total_modes_r_plus_q", int(model["total_modes"])),
                    ("kernel", kernel),
                    ("epsilon", epsilon),
                    ("noise", model["noise"]),
                    ("signal_variance", signal_variance),
                    ("gpr_training_objective", model.get("gpr_training_objective", "log_marginal_likelihood")),
                    ("negative_log_marginal_likelihood", model.get("negative_log_marginal_likelihood", np.nan)),
                    ("num_gpr_features", int(model["num_gpr_features"])),
                    ("num_features", int(model.get("num_features", coeffs.shape[1]))),
                    ("num_quadratic_features", int(model.get("num_quadratic_features", 0))),
                    ("num_mixed_features", int(model.get("num_mixed_features", 0))),
                    (
                        "num_secondary_quadratic_features",
                        int(model.get("num_secondary_quadratic_features", 0)),
                    ),
                    ("ridge_c", model["ridge_c"]),
                    ("ridge_a", model["ridge_a"]),
                    ("ridge_h", model["ridge_h"]),
                    ("ridge_gpr", model["ridge_gpr"]),
                    ("include_quadratic", include_quadratic),
                    ("include_full_quadratic", include_full_quadratic),
                    ("dynamics", dynamics),
                    ("regularizer_convention", model.get("regularizer_convention", "opinf_direct_weight")),
                    ("energy_captured_metric", model["energy_captured"]),
                    ("relative_reconstruction_error", model["relative_reconstruction_error"]),
                    ("relative_derivative_error", model["relative_derivative_error"]),
                    ("training_rollout_error_selected", model["training_rollout_error"]),
                ],
            ),
            (
                "rollout",
                [
                    ("full_window_error", full_error),
                    ("training_window_error", train_error),
                    ("prediction_window_error", prediction_error),
                    ("unstable", bool(unstable)),
                    ("unstable_index", int(unstable_index)),
                    ("num_time_samples", int(times.size)),
                    ("rk4_substeps", rk4_substeps),
                ],
            ),
            (
                "uncertainty",
                [
                    ("posterior_std_max", posterior_std_max),
                    ("posterior_std_time_p95", posterior_std_time_p95),
                    ("train_posterior_std_max", train_posterior_std_max),
                    ("prediction_posterior_std_max", prediction_posterior_std_max),
                    (
                        "secondary_95_half_width_max",
                        float(np.nanmax(secondary_p95)) if compute_uncertainty else np.nan,
                    ),
                    ("state_95_half_width_max_over_time", state_p95_max_over_time),
                    ("state_95_half_width_time_p95", state_p95_time_p95),
                    ("uncertainty_diagnostics_computed", bool(compute_uncertainty)),
                    ("uncertainty_used_in_dynamics", False),
                ],
            ),
            (
                "outputs",
                [
                    ("spacetime_png", spacetime_plot),
                    ("snapshots_png", snapshot_plot),
                    ("error_history_png", error_plot),
                    ("uncertainty_png", uncertainty_plot),
                    ("uncertainty_bands_png", uncertainty_bands_plot),
                    ("q_rollout_npz", q_path),
                    ("summary_txt", summary_path),
                ],
            ),
        ],
    )

    print("\n====================================================")
    print("            KDV GPR-NM-MPOD-OPINF ROLLOUT")
    print("====================================================")
    print(f"[KDV-GPR-NM-MPOD] model={model_path}")
    print(f"[KDV-GPR-NM-MPOD] operator_mode={operator_mode}")
    print(f"[KDV-GPR-NM-MPOD] include_quadratic={include_quadratic}")
    print(f"[KDV-GPR-NM-MPOD] include_full_quadratic={include_full_quadratic}")
    print(
        f"[KDV-GPR-NM-MPOD] r={num_modes}, rbar={num_secondary}, kernel={kernel}, eps={epsilon:.3e}, "
        f"noise={float(model['noise']):.3e}, signal_var={signal_variance:.3e}, "
        f"regularizer=(c={float(model['ridge_c']):.3e}, "
        f"A={float(model['ridge_a']):.3e}, H={float(model['ridge_h']):.3e}, "
        f"GPR={float(model['ridge_gpr']):.3e})"
    )
    print(f"[KDV-GPR-NM-MPOD] train error={train_error:.6e}")
    print(f"[KDV-GPR-NM-MPOD] prediction error={prediction_error:.6e}")
    print(f"[KDV-GPR-NM-MPOD] full error={full_error:.6e}")
    if compute_uncertainty:
        print(f"[KDV-GPR-NM-MPOD] posterior std max={posterior_std_max:.6e}")
        print(f"[KDV-GPR-NM-MPOD] state 95% half-width max={state_p95_max_over_time:.6e}")
    print(f"[KDV-GPR-NM-MPOD] summary saved: {summary_path}")
    return rom, full_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPR-NM-MPOD-OpInf for KdV.")
    parser.add_argument("--model-path", default=os.path.join(SCRIPT_DIR, "models", "gpr_nm_mpod_latentclosure_r5_q9.npz"))
    parser.add_argument("--snapshot-file", default=None)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "Results", "OpInf", "GPR-NM-MPOD", "LatentClosure", "r5_q9"))
    parser.add_argument("--max-norm", type=float, default=1e6)
    parser.add_argument("--no-uncertainty", action="store_true", help="Skip GP posterior uncertainty diagnostics and band plots.")
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        snapshot_file=args.snapshot_file,
        results_dir=args.results_dir,
        max_norm=args.max_norm,
        compute_uncertainty=not args.no_uncertainty,
    )
