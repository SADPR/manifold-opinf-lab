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

## Recommended ANN-NM-MPOD Cases

Run the standard linear-subspace quadratic continuous baseline:

```bash
python3 -B OpInf/stage1_fit_standard_continuous_opinf.py \
  --num-modes 20 \
  --model-path OpInf/models/standard_continuous_quadratic_r20.npz \
  --results-dir Results/OpInf-Standard/Continuous/Training/r20 \
  --ridges 1e-4,1e-2,1e0,1e2,1e4 \
  --rk4-substeps 1,2,5,10

python3 -B OpInf/run_standard_continuous_opinf_suite.py \
  --model-path OpInf/models/standard_continuous_quadratic_r20.npz \
  --results-dir Results/OpInf-Standard/Continuous/r20
```

Train the shared ANN decoder:

```bash
python3 -B OpInf/stage1_fit_ann_manifold_opinf.py \
  --num-primary 20 \
  --num-secondary 123 \
  --hidden-dims 32,64,128,256,256 \
  --model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Manifold/Training/r20_q123
```

Continuous-time paper-faithful case:

```bash
python3 -B OpInf/stage2_fit_ann_continuous_tuned_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --model-path OpInf/models/ann_nm_mpod_noquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/NoQuadratic/Training/r20_q123 \
  --ridges 1e-4,1e-2,1e0,1e2,1e4,1e6 \
  --rk4-substeps 1,2,5,10

python3 -B OpInf/run_ann_manifold_opinf_suite.py \
  --model-path OpInf/models/ann_nm_mpod_noquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/NoQuadratic/r20_q123
```

For a quadratic PDE like Burgers, the physics-consistent operator library can
include compact quadratic state features. The no-quadratic ANN-NM-MPOD command
above is the best empirical continuous result so far and should be described as
an ablation/stabilized variant. The quadratic ANN-NM-MPOD variant, polynomial
MPOD baselines, and the standard linear-subspace baseline are documented in
`OpInf/README.md`.

Discrete-time case:

```bash
python3 -B OpInf/stage2_fit_ann_discrete_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --model-path OpInf/models/ann_manifold_discrete_delta_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Discrete/Training/r20_q123 \
  --ridges 1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2

python3 -B OpInf/run_ann_discrete_opinf_suite.py \
  --model-path OpInf/models/ann_manifold_discrete_delta_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/Discrete/r20_q123
```

See `OpInf/README.md` for details, one-case commands, and oracle diagnostics.
The same README also documents the polynomial MPOD no-quadratic baseline and
the Burgers-specific quadratic ANN-NM-MPOD extension.
