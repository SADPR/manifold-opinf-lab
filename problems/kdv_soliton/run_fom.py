#!/usr/bin/env python3
"""Generate the KdV soliton FOM snapshots from Geelen et al."""

import argparse
import os
import time

from kdv.config import (
    ALPHA,
    BETA,
    DEALIAS,
    DT,
    ETDRK4_CONTOUR_POINTS,
    FINAL_TIME,
    NUM_POINTS,
    TRAIN_FINAL_TIME,
    XL,
    XU,
)
from kdv.core import (
    plot_invariant_drift,
    plot_snapshots,
    plot_spacetime,
    save_snapshot_npz,
    solve_kdv_etdrk4,
    write_fom_report,
)


def main(
    results_dir=os.path.join("Results", "FOM"),
    snapshot_file=None,
    dt=DT,
    final_time=FINAL_TIME,
    train_final_time=TRAIN_FINAL_TIME,
    num_points=NUM_POINTS,
    alpha=ALPHA,
    beta=BETA,
    contour_points=ETDRK4_CONTOUR_POINTS,
    dealias=DEALIAS,
    force=False,
    no_plots=False,
):
    os.makedirs(results_dir, exist_ok=True)
    if snapshot_file is None:
        snapshot_file = os.path.join(results_dir, "kdv_soliton_fom_snapshots.npz")
    report_file = os.path.join(results_dir, "kdv_soliton_fom_summary.txt")
    spacetime_plot = os.path.join(results_dir, "kdv_soliton_fom_spacetime.png")
    snapshots_plot = os.path.join(results_dir, "kdv_soliton_fom_snapshots.png")
    invariant_plot = os.path.join(results_dir, "kdv_soliton_fom_invariants.png")

    if os.path.exists(snapshot_file) and not force:
        print(f"[KDV-FOM] Snapshot file already exists: {snapshot_file}")
        print("[KDV-FOM] Use --force to recompute.")
        return snapshot_file

    print("\n====================================================")
    print("                KDV SOLITON FOM")
    print("====================================================")
    print("[KDV-FOM] PDE: s_t = -alpha*s*s_x - beta*s_xxx")
    print(f"[KDV-FOM] alpha={float(alpha):.6g}, beta={float(beta):.6g}")
    print(f"[KDV-FOM] domain=[{XL:.6g}, {XU:.6g}), periodic")
    print(f"[KDV-FOM] num_points={int(num_points)}, dt={float(dt):.6g}, final_time={float(final_time):.6g}")
    print(f"[KDV-FOM] training window: [0, {float(train_final_time):.6g}]")

    start = time.time()
    x, times, snaps = solve_kdv_etdrk4(
        num_points=num_points,
        dt=dt,
        final_time=final_time,
        alpha=alpha,
        beta=beta,
        contour_points=contour_points,
        dealias=dealias,
    )
    elapsed = time.time() - start
    print(f"[KDV-FOM] snapshots shape={snaps.shape}, elapsed={elapsed:.3e}s")

    save_snapshot_npz(
        snapshot_file,
        x,
        times,
        snaps,
        train_final_time=train_final_time,
        alpha=float(alpha),
        beta=float(beta),
        dt=float(dt),
        final_time=float(final_time),
        num_points=int(num_points),
        xl=float(XL),
        xu=float(XU),
        method="Fourier pseudospectral ETDRK4",
        dealias=int(bool(dealias)),
    )
    write_fom_report(
        report_file,
        snapshot_file,
        x,
        times,
        snaps,
        dt=dt,
        final_time=final_time,
        train_final_time=train_final_time,
        alpha=alpha,
        beta=beta,
        dealias=dealias,
    )

    if not no_plots:
        plot_spacetime(x, times, snaps, spacetime_plot, train_final_time=train_final_time)
        plot_snapshots(x, times, snaps, snapshots_plot, requested_times=(0.0, train_final_time, final_time))
        plot_invariant_drift(times, snaps, invariant_plot)
        print(f"[KDV-FOM] spacetime plot: {spacetime_plot}")
        print(f"[KDV-FOM] snapshots plot: {snapshots_plot}")
        print(f"[KDV-FOM] invariant plot: {invariant_plot}")

    print(f"[KDV-FOM] snapshot file: {snapshot_file}")
    print(f"[KDV-FOM] summary: {report_file}")
    return snapshot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KdV soliton FOM snapshots.")
    parser.add_argument("--results-dir", default=os.path.join("Results", "FOM"))
    parser.add_argument("--snapshot-file", default=None)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--train-final-time", type=float, default=TRAIN_FINAL_TIME)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--contour-points", type=int, default=ETDRK4_CONTOUR_POINTS)
    parser.add_argument("--no-dealias", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        snapshot_file=args.snapshot_file,
        dt=args.dt,
        final_time=args.final_time,
        train_final_time=args.train_final_time,
        num_points=args.num_points,
        alpha=args.alpha,
        beta=args.beta,
        contour_points=args.contour_points,
        dealias=not args.no_dealias,
        force=args.force,
        no_plots=args.no_plots,
    )

