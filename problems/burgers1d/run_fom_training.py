#!/usr/bin/env python3
"""Generate or load the 9 training HDM snapshot sets for 1D Burgers."""

import os
import time
from datetime import datetime

import numpy as np

from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, TIME_SCHEME, U0
from burgers.core import get_snapshot_params, load_or_compute_snaps, param_to_snap_fn


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
    snap_folder=os.path.join("Results", "param_snaps"),
    report_file=os.path.join("Results", "FOMTraining", "fom_training_summary.txt"),
    metadata_file=os.path.join("Results", "FOMTraining", "fom_training_metadata.npz"),
    dt=DT,
    num_steps=NUM_STEPS,
):
    os.makedirs(snap_folder, exist_ok=True)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    mu_list = get_snapshot_params()
    if not mu_list:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    print("\n====================================================")
    print("          1D FOM TRAINING SNAPSHOT GENERATION")
    print("====================================================")
    print(f"[FOM-TRAIN] Number of training parameters: {len(mu_list)}")
    print(f"[FOM-TRAIN] Number of cells: {NUM_CELLS}")

    params = []
    elapsed_list = []
    cached_flags = []
    snapshot_shapes = []

    t_total0 = time.time()
    for mu in mu_list:
        mu = [float(mu[0]), float(mu[1])]
        snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
        was_cached = os.path.exists(snap_fn)

        t0 = time.time()
        snaps = np.asarray(
            load_or_compute_snaps(mu, GRID_X, U0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        elapsed = time.time() - t0

        params.append(mu)
        elapsed_list.append(float(elapsed))
        cached_flags.append(bool(was_cached))
        snapshot_shapes.append(list(snaps.shape))

        status = "cache" if was_cached else "computed"
        print(
            f"[FOM-TRAIN] mu=({mu[0]:.3f}, {mu[1]:.4f}) | {status} | "
            f"shape={snaps.shape} | time={elapsed:.3e}s"
        )

    elapsed_total = time.time() - t_total0
    params_arr = np.asarray(params, dtype=np.float64)
    elapsed_arr = np.asarray(elapsed_list, dtype=np.float64)
    cached_arr = np.asarray(cached_flags, dtype=np.int64)
    shapes_arr = np.asarray(snapshot_shapes, dtype=np.int64)

    n_cached = int(np.sum(cached_arr))
    n_computed = int(cached_arr.size - n_cached)

    np.savez(
        metadata_file,
        params=params_arr,
        elapsed_seconds=elapsed_arr,
        was_cached=cached_arr,
        snapshot_shapes=shapes_arr,
        dt=np.asarray(float(dt), dtype=np.float64),
        num_steps=np.asarray(int(num_steps), dtype=np.int64),
        num_cells=np.asarray(int(NUM_CELLS), dtype=np.int64),
        time_scheme=np.asarray(str(TIME_SCHEME)),
    )

    write_txt_report(
        report_file,
        [
            ("run", [("timestamp", datetime.now().isoformat(timespec="seconds")), ("script", "run_fom_training.py")]),
            (
                "configuration",
                [
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("num_cells", NUM_CELLS),
                    ("time_scheme", TIME_SCHEME),
                    ("num_training_parameters", len(mu_list)),
                ],
            ),
            (
                "results",
                [
                    ("num_loaded_from_cache", n_cached),
                    ("num_computed_new", n_computed),
                    ("total_time_seconds", elapsed_total),
                    ("mean_time_per_parameter_seconds", float(np.mean(elapsed_arr))),
                    ("snapshot_shape_example", shapes_arr[0].tolist() if shapes_arr.size > 0 else None),
                ],
            ),
            ("outputs", [("metadata_npz", metadata_file), ("summary_txt", report_file)]),
        ],
    )
    print(f"[FOM-TRAIN] Metadata saved: {metadata_file}")
    print(f"[FOM-TRAIN] Summary saved: {report_file}")
    return elapsed_total, n_computed


if __name__ == "__main__":
    main()
