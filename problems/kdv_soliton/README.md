# KdV Soliton Benchmark

This problem reproduces the FOM setup from the Korteweg-de Vries experiment in
Geelen, Balzano, Wright, and Willcox, "Learning physics-based reduced-order
models from data using nonlinear manifolds".

The PDE is

```text
s_t = -alpha s s_x - beta s_xxx
```

with periodic boundary conditions on `[-pi, pi)`.

Default settings:

- `alpha = 4`
- `beta = 1`
- `num_points = 256`
- `dt = 0.0002`
- `T = 1`
- training window `t in [0, 0.2]`
- initial condition `s0(x) = 1 + 24 sech^2(sqrt(8) x)`

The solver is Fourier pseudospectral in space and ETDRK4 in time. The linear
dispersive term is handled exactly by the exponential time stepper.

## Run FOM

```bash
cd /home/kratos/manifold-opinf-lab/problems/kdv_soliton
python3 -B run_fom.py
```

Outputs:

- `Results/FOM/kdv_soliton_fom_snapshots.npz`
- `Results/FOM/kdv_soliton_fom_summary.txt`
- `Results/FOM/kdv_soliton_fom_spacetime.png`
- `Results/FOM/kdv_soliton_fom_snapshots.png`
- `Results/FOM/kdv_soliton_fom_invariants.png`

The `.npz` file contains:

- `x`: periodic spatial grid,
- `times`: saved time coordinates,
- `snapshots`: state matrix with shape `(num_points, num_saved_times)`,
- `train_mask`: boolean-like mask selecting `t <= 0.2`.

## Standard OpInf

The first baseline from the paper is standard linear-subspace quadratic OpInf:

```text
s(t) ~= s_ref + V q(t)
dq/dt = c + A q + H q_quad
```

Here "linear-subspace" refers to the state approximation. The reduced dynamics
are quadratic because the KdV nonlinearity `s s_x` is quadratic in the state.
This baseline uses only the first `r` POD modes. It does not use `Vbar`,
polynomial manifold corrections, or alternating minimization.

Fit the `r = 5` model:

```bash
python3 -B OpInf/stage1_fit_standard_opinf.py \
  --num-modes 5
```

The default training grid uses separate direct Tikhonov-weight blocks for the
constant, linear, and quadratic operators. These are `opinf`-style regularizer
weights: the objective contains `||Gamma O^T||_F^2`.

Run the rollout from `t = 0` to `T = 1`:

```bash
python3 -B OpInf/run_standard_opinf.py \
  --model-path OpInf/models/standard_quadratic_opinf_r5.npz \
  --results-dir Results/OpInf/Standard/r5
```

Outputs:

- `OpInf/models/standard_quadratic_opinf_r5.npz`
- `Results/OpInf/Training/standard_opinf_r5_training_summary.txt`
- `Results/OpInf/Standard/r5/standard_opinf_r5_summary.txt`
- `Results/OpInf/Standard/r5/standard_opinf_r5_spacetime.png`
- `Results/OpInf/Standard/r5/standard_opinf_r5_snapshots.png`
- `Results/OpInf/Standard/r5/standard_opinf_r5_error_history.png`

## MPOD-OpInf

The POD-based polynomial manifold model from the paper uses

```text
s(t) ~= s_ref + V q(t) + Vbar Xi g(q(t))
g(q) = [q^2; ...; q^p]
dq/dt = c + A q + H q_quad + P ghat(q)
```

For the KdV benchmark in the paper, the reported setup is `r = 5`, `p = 2`,
and `r + rbar = 14`, so the number of secondary POD modes is `rbar = 9`.
Some filenames still use the historical token `q9`; read it as `rbar=9`, not
as the evolved primary vector `q(t)`.

Fit the MPOD-OpInf model:

```bash
python3 -B OpInf/stage1_fit_mpod_opinf.py \
  --num-modes 5 \
  --total-modes 14 \
  --degree 2 \
  --gamma 1e-3
```

The default grid uses three Tikhonov-weight blocks:

- one weight for `c` and `A`,
- one weight for `H`,
- one weight for the higher polynomial block `P`.

These are direct `opinf`-style regularizer weights: the objective contains
`||Gamma O^T||_F^2`, so a weight `1e4` enters as `1e8 ||O||^2`.

Run the MPOD rollout:

```bash
python3 -B OpInf/run_mpod_opinf.py \
  --model-path OpInf/models/mpod_opinf_r5_p2_q9.npz \
  --results-dir Results/OpInf/MPOD/r5_p2_q9
```

Current reference output from this implementation:

```text
training window error:   4.265206e-01
prediction window error: 4.479676e-01
full window error:       4.438480e-01
```

Outputs:

- `OpInf/models/mpod_opinf_r5_p2_q9.npz`
- `Results/OpInf/Training/mpod_opinf_r5_p2_q9_training_summary.txt`
- `Results/OpInf/Training/mpod_opinf_r5_p2_q9_ridge_grid.csv`
- `Results/OpInf/MPOD/r5_p2_q9/mpod_opinf_r5_p2_q9_summary.txt`
- `Results/OpInf/MPOD/r5_p2_q9/mpod_opinf_r5_p2_q9_spacetime.png`
- `Results/OpInf/MPOD/r5_p2_q9/mpod_opinf_r5_p2_q9_snapshots.png`
- `Results/OpInf/MPOD/r5_p2_q9/mpod_opinf_r5_p2_q9_error_history.png`

## RBF-NM-MPOD-OpInf

Here `NM-MPOD` means **Nonlinear-Map Manifold POD**: we keep the MPOD
primary/secondary POD split, but replace the polynomial secondary-coordinate
map with a learned nonlinear map.

This is an experimental Nonlinear-Map MPOD variant. It keeps the POD primary and
secondary bases,

```text
s(t) ~= s_ref + V q(t) + Vbar W phi_RBF(q(t))
dq/dt = c + A q + H q_quad + P phi_RBF(q)
```

For the first `r = 5` trial, we use the paper's same total dimension
`r + rbar = 14`, so the secondary dimension is `rbar = 9`. The RBF map uses the full
set of training reduced coordinates as centers by default. This is not the same
closed polynomial algebra as the paper's MPOD derivation; it is a practical
nonlinear-feature analog where the polynomial map is replaced by an RBF map.

Fit the RBF-NM-MPOD-OpInf model:

```bash
python3 -B OpInf/stage1_fit_rbf_nm_mpod_opinf.py \
  --num-modes 5 \
  --total-modes 14 \
  --kernels imq,gaussian \
  --epsilons 0.01,0.013869189,0.019235439,0.026677993,0.037000212,0.051316292,0.071171532,0.09870914,0.13690157,0.18987137,0.26333618,0.36522591,0.5065387,0.70252808,0.97434944,1.3513436,1.8742039,2.5993688,3.6051136,5 \
  --rbf-ridges 1e-10,1e-8,1e-6 \
  --regularizer-ca-candidates 1e0 \
  --regularizer-h-candidates 1e4 \
  --regularizer-rbf-candidates 1e4,1e6 \
  --model-path OpInf/models/rbf_nm_mpod_opinf_r5_q9.npz \
  --results-dir Results/OpInf/Training/rbf_nm_mpod_r5_q9
```

Run the rollout:

```bash
python3 -B OpInf/run_rbf_nm_mpod_opinf.py \
  --model-path OpInf/models/rbf_nm_mpod_opinf_r5_q9.npz \
  --results-dir Results/OpInf/RBF-NM-MPOD/r5_q9
```

Current output from this implementation:

```text
selected kernel:          gaussian
selected epsilon:         9.743494e-01
selected RBF ridge:       1.000000e-08
number of RBF centers:    1001
manifold recon. error:    8.574985e-02
training window error:    8.925803e-02
prediction window error:  1.655535e-01
full window error:        1.536274e-01
```

Outputs:

- `OpInf/models/rbf_nm_mpod_opinf_r5_q9.npz`
- `Results/OpInf/Training/rbf_nm_mpod_r5_q9/rbf_nm_mpod_opinf_r5_q9_training_summary.txt`
- `Results/OpInf/Training/rbf_nm_mpod_r5_q9/rbf_nm_mpod_opinf_r5_q9_grid.csv`
- `Results/OpInf/RBF-NM-MPOD/r5_q9/rbf_nm_mpod_opinf_r5_q9_summary.txt`
- `Results/OpInf/RBF-NM-MPOD/r5_q9/rbf_nm_mpod_opinf_r5_q9_spacetime.png`
- `Results/OpInf/RBF-NM-MPOD/r5_q9/rbf_nm_mpod_opinf_r5_q9_snapshots.png`
- `Results/OpInf/RBF-NM-MPOD/r5_q9/rbf_nm_mpod_opinf_r5_q9_error_history.png`

## GPR-NM-MPOD-OpInf

This is a second experimental Nonlinear-Map MPOD variant. It uses a Gaussian-process
posterior mean to predict the secondary coordinates directly,

```text
s(t) ~= s_ref + V q(t) + Vbar z_GPR(q(t))
dq/dt = c + A q + H q_quad + B z_GPR(q)
```

Unlike the raw RBF-feature model above, the online nonlinear feature block has
dimension `rbar = 9`, not one feature per training center. For `r = 5`, the learned
operator therefore has `1 + r + r(r+1)/2 + rbar = 30` features. The GP
hyperparameters are selected by maximizing the multi-output log marginal
likelihood; the online model then uses only the posterior mean `z_GPR(q)`.

Fit the GPR-NM-MPOD-OpInf model:

```bash
python3 -B OpInf/stage1_fit_gpr_nm_mpod_opinf.py \
  --num-modes 5 \
  --total-modes 14 \
  --kernels gaussian,matern32 \
  --initial-epsilon 0.5 \
  --initial-noise 1e-6 \
  --initial-signal-variance 1.0 \
  --epsilon-bounds 1e-3,10 \
  --noise-bounds 1e-12,1e-2 \
  --signal-variance-bounds 1e-6,1e6 \
  --gpr-optimizer-maxiter 60 \
  --regularizer-ca-candidates 1e0 \
  --regularizer-h-candidates 1e4 \
  --regularizer-gpr-candidates 1e0,1e2,1e4 \
  --model-path OpInf/models/gpr_nm_mpod_opinf_r5_q9.npz \
  --results-dir Results/OpInf/Training/gpr_nm_mpod_r5_q9
```

Run the rollout:

```bash
python3 -B OpInf/run_gpr_nm_mpod_opinf.py \
  --model-path OpInf/models/gpr_nm_mpod_opinf_r5_q9.npz \
  --results-dir Results/OpInf/GPR-NM-MPOD/r5_q9
```

The rollout computes GP uncertainty diagnostics by default. Use
`--no-uncertainty` to skip the posterior variance and 95% band plots.

Current output from this implementation:

```text
selected kernel:          matern32
selected epsilon:         7.843971e-03
selected GP noise:        4.754949e-12
selected signal variance: 1.000000e+06
negative log marginal ML: -2.992483e+04
manifold recon. error:    8.575158e-02
training window error:    8.575197e-02
prediction window error:  8.770054e-02
full window error:        8.732352e-02
posterior std max:        2.090220e-02
state 95 half-width max:  8.250215e-03
```

The uncertainty is diagnostic only: it is not fed back into the ODE. The most
didactic plot is the snapshot band plot. Its first row shows the physical
solution scale, where the band may be visually thin; its second row shows the
residual `Reference - GPR-NM-MPOD` with the GP-induced 95% band on the same error
scale; its third row shows the GP band on its own scale.

Outputs:

- `OpInf/models/gpr_nm_mpod_opinf_r5_q9.npz`
- `Results/OpInf/Training/gpr_nm_mpod_r5_q9/gpr_nm_mpod_opinf_r5_q9_training_summary.txt`
- `Results/OpInf/Training/gpr_nm_mpod_r5_q9/gpr_nm_mpod_opinf_r5_q9_regularizer_grid.csv`
- `Results/OpInf/GPR-NM-MPOD/r5_q9/gpr_nm_mpod_opinf_r5_q9_summary.txt`
- `Results/OpInf/GPR-NM-MPOD/r5_q9/gpr_nm_mpod_opinf_r5_q9_spacetime.png`
- `Results/OpInf/GPR-NM-MPOD/r5_q9/gpr_nm_mpod_opinf_r5_q9_snapshots.png`
- `Results/OpInf/GPR-NM-MPOD/r5_q9/gpr_nm_mpod_opinf_r5_q9_error_history.png`
- `Results/OpInf/GPR-NM-MPOD/r5_q9/gpr_nm_mpod_opinf_r5_q9_uncertainty.png`
- `Results/OpInf/GPR-NM-MPOD/r5_q9/gpr_nm_mpod_opinf_r5_q9_uncertainty_bands.png`

## Smoke Test

For a fast low-resolution check:

```bash
python3 -B run_fom.py \
  --num-points 64 \
  --dt 0.001 \
  --final-time 0.01 \
  --train-final-time 0.005 \
  --results-dir Results/FOM_smoke \
  --force
```
