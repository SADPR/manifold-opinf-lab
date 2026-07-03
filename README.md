# manifold-opinf-lab

Laboratory repository for nonlinear-manifold Operator Inference experiments.

The goal is to keep the OpInf/manifold research code separate from the
problem-specific PROM/HPROM workbenches. Each benchmark problem should be
self-contained under `problems/`, with only the FOM snapshot generation, POD
basis construction, OpInf training, and OpInf evaluation code needed for that
problem.

Shared equation-agnostic helpers live in `opinf_lab/`. They cover reduced
feature construction, POD projection/reconstruction, regularized least-squares
OpInf solves, and RK4 rollout. Problem directories should keep only
problem-specific snapshot loading, parameter grids, plotting, and command-line
workflows.

## Current Layout

```text
opinf_lab/                   # shared reduced OpInf helper package
problems/
  burgers1d/
    burgers/                 # minimal 1D Burgers FOM/POD utilities
    POD/                     # POD basis generation and diagnostics
    OpInf/                   # Burgers-specific OpInf experiment scripts
    run_fom_training.py      # training snapshot generation
  kdv_soliton/
    kdv/                     # Fourier ETDRK4 KdV FOM utilities
    run_fom.py               # KdV snapshot generation
```

Generated data are intentionally ignored by git:

- `problems/*/Results/`
- `problems/*/POD/*.npy`
- `problems/*/POD/*.npz`
- `problems/*/POD/*.png`
- `problems/*/POD/*.txt`
- `problems/*/OpInf/models/`

## Burgers 1D

Run from the Burgers problem directory:

```bash
cd /home/kratos/manifold-opinf-lab/problems/burgers1d
```

Build the FOM training snapshots and POD basis:

```bash
python3 -B run_fom_training.py
python3 -B POD/stage1_build_pod_basis.py
```

Train the shared ANN manifold decoder:

```bash
python3 -B OpInf/stage1_fit_ann_manifold_opinf.py \
  --num-primary 20 \
  --num-secondary 123 \
  --hidden-dims 32,64,128,256,256 \
  --model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Manifold/Training/r20_q123
```

Case 1, continuous-time ANN-NM-MPOD OpInf:

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

Case 2, discrete-time ANN-NM-MPOD OpInf:

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

See `problems/burgers1d/OpInf/README.md` for the detailed Burgers-specific
workflow, the standard linear-subspace quadratic baseline, polynomial MPOD
baselines, and oracle diagnostics.

## KdV Soliton

The Korteweg-de Vries soliton benchmark from Geelen, Balzano, Wright, and
Willcox is available under `problems/kdv_soliton/`:

```text
s_t = -alpha s s_x - beta s_xxx
x in [-pi, pi], periodic
train t in [0, 0.2], predict t in [0.2, 1]
```

Run the FOM:

```bash
cd /home/kratos/manifold-opinf-lab/problems/kdv_soliton
python3 -B run_fom.py
```

The implemented KdV ROMs are:

- standard linear-subspace quadratic OpInf,
- polynomial MPOD-OpInf,
- RBF-NM-MPOD-OpInf,
- GPR-NM-MPOD-OpInf,
- alternating-minimization MAM-OpInf,

Run the standard linear-subspace quadratic OpInf baseline:

```bash
cd /home/kratos/manifold-opinf-lab/problems/kdv_soliton
python3 -B OpInf/stage1_fit_standard_opinf.py --num-modes 5
python3 -B OpInf/run_standard_opinf.py \
  --model-path OpInf/models/standard_quadratic_opinf_r5.npz \
  --results-dir Results/OpInf/Standard/r5
```

See `problems/kdv_soliton/README.md` for the MPOD, RBF-NM-MPOD, GPR-NM-MPOD,
their no-quadratic ablations, MAM commands, and current reproduction status.

Here `NM-MPOD` means **Nonlinear-Map Manifold POD**: an MPOD-style secondary
manifold correction whose secondary-coordinate map is learned by RBF, GPR, ANN,
or another nonlinear regressor.
