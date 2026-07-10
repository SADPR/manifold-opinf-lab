# Burgers 1D OpInf Benchmark

This directory contains the minimal 1D Burgers pieces needed for manifold
Operator Inference experiments:

- `burgers/`: FOM solver, snapshot cache helpers, POD utilities, plotting.
- `POD/`: POD basis generation and diagnostics.
- `OpInf/`: linear, polynomial-manifold, RBF-manifold, and ANN-NM-MPOD OpInf
  experiments.
- `run_fom_training.py`: training snapshot generation.

The PROM/HPROM infrastructure from `burgers1d-rom-workbench` is intentionally
not copied here.

## Rebuild Data

Run from this directory:

```bash
cd /home/kratos/manifold-opinf-lab/problems/burgers1d
python3 -B run_fom_training.py
python3 -B POD/stage1_build_pod_basis.py
```

## Recommended Continuous OpInf Cases

Run the standard linear-subspace quadratic continuous baseline:

```bash
python3 -B OpInf/stage1_fit_standard_continuous_opinf.py \
  --num-modes 10 \
  --model-path OpInf/models/standard_continuous_quadratic_r10.npz \
  --results-dir Results/OpInf-Standard/Continuous/Training/r10 \
  --ridges 1e-4,1e-2,1e0,1e2,1e4 \
  --rk4-substeps 1,2,5,10

python3 -B OpInf/run_standard_continuous_opinf_suite.py \
  --model-path OpInf/models/standard_continuous_quadratic_r10.npz \
  --results-dir Results/OpInf-Standard/Continuous/r10
```

Run the polynomial MPOD baseline with `r=10` primary modes and `rbar=133`
secondary modes. `stage1_fit_manifold_opinf.py` fits the decoder and a single
fixed-ridge dynamics operator (quick, but not tuned); prefer
`stage2_fit_manifold_continuous_tuned_opinf.py`, which recomputes the decoder
and then runs the same ridge x RK4-substep grid + held-out-trajectory
validation used for the ANN models below, so MPOD and NM-MPOD are compared
on equal tuning footing:

```bash
python3 -B OpInf/stage2_fit_manifold_continuous_tuned_opinf.py \
  --num-primary 10 \
  --num-secondary 133 \
  --polynomial-order 2 \
  --model-path OpInf/models/mpod_induced_continuous_tuned_r10_q133_p2.npz \
  --results-dir Results/OpInf-MPOD-Induced/ContinuousTuned/Training

python3 -B OpInf/run_manifold_opinf_suite.py \
  --model-path OpInf/models/mpod_induced_continuous_tuned_r10_q133_p2.npz \
  --results-dir Results/OpInf-MPOD-Induced/r10_q133_p2_tuned
```

`--polynomial-order` can be raised (e.g. `4`, `5`) to sweep MPOD's induced
higher-order library; watch for `RuntimeWarning: overflow` during the sweep --
this library is genuinely fragile at low ridge (see the report in
`../../reports/nm_mpod_proposal/`), and the script's full-data-refit stability
check exists specifically to keep that fragility from silently picking an
unstable "best" candidate.

Train the shared ANN decoder and then one of the three AL/PQ/QC dynamics
libraries in the report's operator hierarchy (`--operator-mode al|pq|qc`):

```bash
python3 -B OpInf/stage1_fit_ann_manifold_opinf.py \
  --num-primary 10 \
  --num-secondary 133 \
  --hidden-dims 32,64,128,256,256 \
  --model-path OpInf/models/ann_manifold_linear_plus_ann_r10_q133.npz \
  --results-dir Results/OpInf-ANN-Manifold/Training/r10_q133

python3 -B OpInf/stage2_fit_ann_continuous_tuned_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r10_q133.npz \
  --operator-mode qc \
  --model-path OpInf/models/ann_nm_mpod_qc_continuous_tuned_r10_q133.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/ContinuousTuned/Training \
  --ridges 1e-2,1e0,1e2,1e4,1e6 \
  --rk4-substeps 1,2

python3 -B OpInf/run_ann_manifold_opinf_suite.py \
  --model-path OpInf/models/ann_nm_mpod_qc_continuous_tuned_r10_q133.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/QC/r10_q133
```

Note that `stage1_fit_ann_manifold_opinf.py`'s validation split holds out
whole trajectories (not scattered snapshots), so it needs `group_ids`; if you
call `train_ann_secondary_map` directly rather than through this script,
build the trajectory-index array the same way this script does or you will
get a leaky, over-optimistic validation curve.

Current suite errors (all rows tuned by the same ridge x substep + held-out
validation protocol; see the report for the extrapolation point and online
wall-clock cost, which is not negligible for QC):

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| Standard quadratic OpInf, `r=10` | 9.11e-02 | 9.11e-02 | 8.74e-02 |
| Induced MPOD (tuned), `r=10,rbar=133,p=2` | 8.19e-02 | 8.39e-02 | 8.12e-02 |
| ANN AL-NM-MPOD, `r=10,rbar=133` | 1.14e-02 | 9.01e-03 | 6.52e-03 |
| ANN PQ-NM-MPOD, `r=10,rbar=133` | 1.37e-02 | 1.33e-02 | 1.32e-02 |
| ANN QC-NM-MPOD, `r=10,rbar=133` | 5.85e-03 | 5.79e-03 | 4.96e-03 |

Here `rbar` denotes the number of secondary POD modes in `Vbar`. Some filenames
still use the historical token `q133`; in the theory and plots this is
`\bar r=133`, not the evolved primary coordinate vector `q(t)`.
