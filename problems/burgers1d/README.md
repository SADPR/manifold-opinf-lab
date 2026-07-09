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
secondary modes:

```bash
python3 -B OpInf/stage1_fit_manifold_opinf.py \
  --num-primary 10 \
  --num-secondary 133 \
  --polynomial-order 2 \
  --dynamics-ridge 1e2 \
  --model-path OpInf/models/mpod_induced_continuous_r10_q133_p2_ridge1e2.npz \
  --results-dir Results/OpInf-MPOD-Induced/Training/r10_q133_p2

python3 -B OpInf/run_manifold_opinf_suite.py \
  --model-path OpInf/models/mpod_induced_continuous_r10_q133_p2_ridge1e2.npz \
  --results-dir Results/OpInf-MPOD-Induced/r10_q133_p2
```

Train the shared ANN decoder and then the full state-quadratic ANN-NM-MPOD
operator:

```bash
python3 -B OpInf/stage1_fit_ann_manifold_opinf.py \
  --num-primary 10 \
  --num-secondary 133 \
  --hidden-dims 32,64,128,256,256 \
  --model-path OpInf/models/ann_manifold_linear_plus_ann_r10_q133.npz \
  --results-dir Results/OpInf-ANN-Manifold/Training/r10_q133

python3 -B OpInf/stage2_fit_ann_continuous_tuned_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r10_q133.npz \
  --model-path OpInf/models/ann_nm_mpod_fullquadratic_continuous_tuned_r10_q133.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/FullQuadratic/Training/r10_q133 \
  --ridges 1e-2,1e0,1e2,1e4,1e6 \
  --rk4-substeps 1,2

python3 -B OpInf/run_ann_manifold_opinf_suite.py \
  --model-path OpInf/models/ann_nm_mpod_fullquadratic_continuous_tuned_r10_q133.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/FullQuadratic/r10_q133
```

Current suite errors:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| Standard quadratic OpInf, `r=10` | 9.11e-02 | 9.11e-02 | 8.74e-02 |
| MPOD, `r=10,rbar=133,p=2` | 8.19e-02 | 8.39e-02 | 8.12e-02 |
| ANN-NM-MPOD full quadratic, `r=10,rbar=133` | 5.97e-03 | 5.76e-03 | 6.56e-03 |

Here `rbar` denotes the number of secondary POD modes in `Vbar`. Some filenames
still use the historical token `q133`; in the theory and plots this is
`\bar r=133`, not the evolved primary coordinate vector `q(t)`.
