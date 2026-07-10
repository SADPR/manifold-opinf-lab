#!/usr/bin/env python3
"""Benchmark online cost (decoder eval, RHS eval, full rollout) for every
Burgers ROM variant, plus a fresh FOM solve for a speedup reference.

The report's original feature-count table implicitly suggested the ANN
decoder was "free" once trained; it is not -- every RHS call pays for a
dense forward pass. This script measures wall-clock cost directly rather
than counting FLOPs, and reports it next to the accuracy numbers so the
comparison in the report is cost-aware, not just error-aware.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ann_manifold_opinf_utils import load_ann_manifold_model, predict_ann_secondary, rhs_ann_continuous, rollout_ann_continuous_rk4
from burgers.config import DT, GRID_X, NUM_CELLS, NUM_STEPS, U0
from burgers.core import inviscid_burgers_implicit1d
from manifold_opinf_utils import (
    continuous_feature_vector,
    elementwise_power_matrix,
    load_manifold_model,
    manifold_decode,
    rhs_continuous,
    rollout_continuous_rk4,
)
from run_standard_continuous_opinf import load_standard_continuous_model as load_standard_model


def _time_calls(fn, n_warmup=5, n_calls=200):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    return (time.perf_counter() - t0) / n_calls


def _time_once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def benchmark_fom(mu, dt, num_steps, repeats=3):
    times = []
    for _ in range(repeats):
        times.append(_time_once(lambda: inviscid_burgers_implicit1d(GRID_X, U0, dt, num_steps, mu, verbose=False)))
    return float(np.median(times))


def benchmark_standard(model_path, q0, mu, dt, num_steps):
    model = load_standard_model(model_path)
    q = np.asarray(q0, dtype=np.float64)

    def rhs_of(qv):
        theta = continuous_feature_vector(
            qv,
            mu,
            feature_mode=model["feature_mode"],
            include_param_linear=bool(model["include_param_linear"]),
            include_quadratic=bool(model["include_quadratic"]),
            include_higher=bool(model["include_higher"]),
            max_degree=int(model["max_degree"]),
        )
        theta_scaled = (theta - model["x_mean"]) / model["x_scale"]
        return model["operator"] @ theta_scaled

    def rhs():
        return rhs_of(q)

    def rollout():
        from opinf_lab.time_integration import rollout_rk4

        return rollout_rk4(rhs_of, q0, dt, num_steps, substeps=int(model.get("rk4_substeps", 1)))

    rhs_cost = _time_calls(rhs, n_calls=500)
    rollout_cost = _time_calls(rollout, n_warmup=1, n_calls=5)
    return {
        "decoder_cost_s": np.nan,
        "rhs_cost_s": rhs_cost,
        "rollout_cost_s": rollout_cost,
        "num_features": int(model["operator"].shape[1]),
    }


def benchmark_mpod(model_path, q0, mu, dt, num_steps):
    model = load_manifold_model(model_path)
    q = np.asarray(q0, dtype=np.float64).reshape(-1, 1)

    def decoder():
        return model["xi"] @ elementwise_power_matrix(q, polynomial_order=int(model["polynomial_order"])).T

    def rhs():
        return rhs_continuous(q0, mu, model)

    def rollout():
        return rollout_continuous_rk4(q0, mu, dt, num_steps, model, substeps=int(model.get("rk4_substeps", 1)))

    decoder_cost = _time_calls(decoder, n_calls=500)
    rhs_cost = _time_calls(rhs, n_calls=500)
    rollout_cost = _time_calls(rollout, n_warmup=1, n_calls=5)
    return {
        "decoder_cost_s": decoder_cost,
        "rhs_cost_s": rhs_cost,
        "rollout_cost_s": rollout_cost,
        "num_features": int(model["operator"].shape[1]),
    }


def benchmark_ann(model_path, q0, mu, dt, num_steps, device=None):
    model = load_ann_manifold_model(model_path, device=device)
    q = np.asarray(q0, dtype=np.float64)

    def decoder():
        return predict_ann_secondary(q, mu, model)

    def rhs():
        return rhs_ann_continuous(q, mu, model)

    def rollout():
        return rollout_ann_continuous_rk4(q0, mu, dt, num_steps, model, substeps=int(model.get("rk4_substeps", 1)))

    decoder_cost = _time_calls(decoder, n_calls=200)
    rhs_cost = _time_calls(rhs, n_calls=200)
    rollout_cost = _time_calls(rollout, n_warmup=1, n_calls=5)
    return {
        "decoder_cost_s": decoder_cost,
        "rhs_cost_s": rhs_cost,
        "rollout_cost_s": rollout_cost,
        "num_features": int(model["operator"].shape[1]),
    }


def main(
    mu=(4.75, 0.020),
    dt=DT,
    num_steps=NUM_STEPS,
    results_path=os.path.join(PROJECT_ROOT, "Results", "online_cost_benchmark.csv"),
    include_fom=True,
):
    mu = (float(mu[0]), float(mu[1]))
    q0_r10 = np.zeros(10, dtype=np.float64)  # RHS cost is state-independent in wall-clock terms; q0 shape only needs to match r

    rows = []
    if include_fom:
        print("[Benchmark] Timing fresh FOM solve (backward Euler + Newton, N=5000)...")
        fom_time = benchmark_fom(mu, dt, num_steps, repeats=3)
        rows.append(("FOM (N=5000, backward Euler)", np.nan, np.nan, fom_time, np.nan, fom_time / fom_time))
        print(f"[Benchmark]   FOM full solve: {fom_time:.4e} s ({num_steps} steps)")

    configs = [
        ("Standard quadratic OpInf, r=10", "standard", os.path.join(SCRIPT_DIR, "models", "standard_continuous_quadratic_r10.npz")),
        ("Induced MPOD, p=2 (tuned)", "mpod", os.path.join(SCRIPT_DIR, "models", "mpod_induced_continuous_tuned_r10_q133_p2.npz")),
        ("Induced MPOD, p=4 (tuned)", "mpod", os.path.join(SCRIPT_DIR, "models", "mpod_induced_continuous_tuned_r10_q133_p4.npz")),
        ("Induced MPOD, p=5 (tuned)", "mpod", os.path.join(SCRIPT_DIR, "models", "mpod_induced_continuous_tuned_r10_q133_p5.npz")),
        ("ANN AL-NM-MPOD", "ann", os.path.join(SCRIPT_DIR, "models", "ann_nm_mpod_al_continuous_tuned_r10_q133.npz")),
        ("ANN PQ-NM-MPOD", "ann", os.path.join(SCRIPT_DIR, "models", "ann_nm_mpod_pq_continuous_tuned_r10_q133.npz")),
        ("ANN QC-NM-MPOD", "ann", os.path.join(SCRIPT_DIR, "models", "ann_nm_mpod_qc_continuous_tuned_r10_q133.npz")),
    ]

    for label, kind, path in configs:
        if not os.path.exists(path):
            print(f"[Benchmark] SKIP {label}: model not found at {path}")
            continue
        print(f"[Benchmark] Timing {label}...")
        if kind == "standard":
            stats = benchmark_standard(path, q0_r10, mu, dt, num_steps)
        elif kind == "mpod":
            stats = benchmark_mpod(path, q0_r10, mu, dt, num_steps)
        elif kind == "ann":
            stats = benchmark_ann(path, q0_r10, mu, dt, num_steps)
        else:
            raise ValueError(kind)
        speedup = (fom_time / stats["rollout_cost_s"]) if include_fom else np.nan
        rows.append((label, stats["decoder_cost_s"], stats["rhs_cost_s"], stats["rollout_cost_s"], stats["num_features"], speedup))
        print(
            f"[Benchmark]   decoder={stats['decoder_cost_s']:.3e} s/call, "
            f"rhs={stats['rhs_cost_s']:.3e} s/call, rollout={stats['rollout_cost_s']:.3e} s "
            f"({int(num_steps)} steps), speedup={speedup:.1f}x"
        )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("model,decoder_cost_s_per_call,rhs_cost_s_per_call,rollout_cost_s,num_rhs_features,speedup_vs_fom\n")
        for label, dcost, rcost, rollcost, nfeat, speedup in rows:
            f.write(f'"{label}",{dcost},{rcost},{rollcost},{nfeat},{speedup}\n')
    print(f"[Benchmark] Saved: {results_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark online cost of Burgers ROM variants.")
    parser.add_argument("--mu1", type=float, default=4.75)
    parser.add_argument("--mu2", type=float, default=0.020)
    parser.add_argument("--results-path", default=os.path.join(PROJECT_ROOT, "Results", "online_cost_benchmark.csv"))
    parser.add_argument("--no-fom", action="store_true")
    args = parser.parse_args()
    main(mu=(args.mu1, args.mu2), results_path=args.results_path, include_fom=not args.no_fom)
