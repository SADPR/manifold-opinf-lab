# OpInf

Experimental Operator Inference ROMs for the 1D Burgers workbench.

The current recommended experiments are the ANN-NM-MPOD models, compared
against standard linear-subspace OpInf and polynomial MPOD baselines:

- `Case 0`: standard continuous linear-subspace quadratic OpInf.
- `Case 1`: continuous-time ANN-NM-MPOD OpInf.
- `Case 2`: discrete-time ANN-NM-MPOD OpInf.

Both cases use the same nonlinear decoder,

```text
u = u_ref + V q + Vbar N_ANN(q, mu),
```

where `q in R^20` are the primary POD coordinates and
`N_ANN(q, mu) in R^123` predicts the secondary POD coordinates. Together this
uses the first `143` POD coordinates from the `1e-4` POD truncation.

Run all commands from the repository root:

```bash
cd /home/kratos/manifold-opinf-lab/problems/burgers1d
```

If the training snapshots and POD basis are not already present, build them
first:

```bash
python3 -B run_fom_training.py
python3 -B POD/stage1_build_pod_basis.py
```

The standard test points used by the suite commands are:

- `(mu1=4.56, mu2=0.019)`
- `(mu1=4.75, mu2=0.020)`
- `(mu1=5.19, mu2=0.026)`

The ridge/grid-search validation inside the training scripts uses held-out
training trajectories, not these three final test points.

## Theory Note

The manifold projection shown in the Willcox-style slide for a linear FOM,

```text
x_dot = A x + B u,
x ~= V q + Vbar g(q),
```

leads after projection with `V^T` to

```text
q_dot = Ahat q + Phat g(q) + Bhat u.
```

There is no extra `H(q kron q)` term in that derivation. The nonlinear ROM
comes from the manifold map `g(q)`.

The Burgers FOM here is not linear. In reduced-coordinate shorthand, the
physics is closer to

```text
x_dot = A x + H(x kron x) + B eta(mu),
```

where `B eta(mu)` represents the parameter-dependent boundary/source forcing.
For Burgers and KdV, the standard linear-subspace OpInf library should
therefore include compact quadratic state features. For nonlinear-manifold
models, we test both:

```text
theta(q, mu) = [1, eta(mu), q, eta(mu) q, q_quad, g(q)]
```

and the no-explicit-quadratic ablation

```text
theta(q, mu) = [1, eta(mu), q, eta(mu) q, g(q)].
```

The no-quadratic ANN-NM-MPOD model performs best here, but it should be
presented as an empirical/stabilized ablation, not as the only physics-faithful
choice for a quadratic PDE.

For the parametric Burgers problem, `mu` is part of the online input. We
therefore condition both the decoder and the reduced dynamics on parameter
features

```text
eta(mu) = [mu1, mu2, mu1^2, mu1 mu2, mu2^2].
```

## Shared ANN Decoder

Train the ANN secondary-coordinate map once:

```bash
python3 -B OpInf/stage1_fit_ann_manifold_opinf.py \
  --num-primary 20 \
  --num-secondary 123 \
  --hidden-dims 32,64,128,256,256 \
  --model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Manifold/Training/r20_q123
```

This writes:

- `OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz`
- `Results/OpInf-ANN-Manifold/Training/r20_q123/`

The stage-1 model also contains an initial continuous operator. For the final
recommended continuous and discrete cases below, we use this file as the ANN
decoder source and then refit/tune the dynamics.

## Case 0: Standard Continuous OpInf

This is the linear-subspace baseline. The decoder is only POD,

```text
u(t) = u_ref + V q(t),
```

and the continuous reduced dynamics include the Burgers-consistent quadratic
state features:

```text
theta(q, mu) = [1, eta(mu), q, eta(mu) q, q_quad].
```

Train/tune and test:

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

Current standard linear-subspace rollout errors:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| standard continuous OpInf, r=20 | 2.78e-01 | 2.73e-01 | 2.42e-01 |

## Case 1: Continuous ANN-NM-MPOD

This is the more theory-aligned OpInf case. It learns a continuous-time ODE and
rolls it out with RK4:

```text
u(t) = u_ref + V q(t) + Vbar N_ANN(q(t), mu)
dq/dt = W theta(q(t), mu)
theta(q, mu) = [1, eta(mu), q, eta(mu) q, N_ANN(q, mu)]
```

There is no explicit quadratic state term `q kron q`. This is the primary
slide-faithful ANN-NM-MPOD case.

Train/tune the continuous operator:

```bash
python3 -B OpInf/stage2_fit_ann_continuous_tuned_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --model-path OpInf/models/ann_nm_mpod_noquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/NoQuadratic/Training/r20_q123 \
  --ridges 1e-4,1e-2,1e0,1e2,1e4,1e6 \
  --rk4-substeps 1,2,5,10
```

Run the three standard test points:

```bash
python3 -B OpInf/run_ann_manifold_opinf_suite.py \
  --model-path OpInf/models/ann_nm_mpod_noquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/NoQuadratic/r20_q123
```

Run one point manually:

```bash
python3 -B OpInf/run_ann_manifold_opinf.py \
  --model-path OpInf/models/ann_nm_mpod_noquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/NoQuadratic/one_case \
  --mu1 4.56 \
  --mu2 0.019
```

For the current run, the selected ridge was `1e-2`, the selected RK4 substeps
were `10`, and the standard rollout errors were approximately:

| mu1 | mu2 | relative state error |
| --- | --- | --- |
| 4.56 | 0.019 | 8.61e-03 |
| 4.75 | 0.020 | 8.73e-03 |
| 5.19 | 0.026 | 8.00e-03 |

Outputs are written under:

- `OpInf/models/ann_nm_mpod_noquadratic_continuous_tuned_r20_q123.npz`
- `Results/OpInf-ANN-NM-MPOD/NoQuadratic/Training/r20_q123/`
- `Results/OpInf-ANN-NM-MPOD/NoQuadratic/r20_q123/`

## Quadratic ANN-NM-MPOD Variant

For Burgers, we also test the physics-motivated quadratic ANN-NM-MPOD variant

```text
theta(q, mu) = [1, eta(mu), q, eta(mu) q, q_quad, N_ANN(q, mu)].
```

This is not the direct linear-FOM manifold projection from the slide. It is the
quadratic-PDE operator-library variant.

```bash
python3 -B OpInf/stage2_fit_ann_continuous_tuned_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --model-path OpInf/models/ann_nm_mpod_quadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/Quadratic/Training/r20_q123 \
  --ridges 1e-4,1e-2,1e0,1e2,1e4,1e6 \
  --rk4-substeps 1,2,5,10 \
  --with-quadratic

python3 -B OpInf/run_ann_manifold_opinf_suite.py \
  --model-path OpInf/models/ann_nm_mpod_quadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/Quadratic/r20_q123
```

In the current Burgers runs, this variant is worse than the no-quadratic ANN
ablation:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| ANN-NM-MPOD, no quadratic | 8.61e-03 | 8.73e-03 | 8.00e-03 |
| ANN-NM-MPOD, quadratic extension | 1.11e-02 | 1.39e-02 | 5.11e-02 |

## Full Quadratic ANN-NM-MPOD Variant

For a quadratic FOM and nonlinear decoder

```text
u(t) = u_ref + V q(t) + Vbar N_ANN(q(t), mu),
```

the fully expanded quadratic state library contains the primary-primary,
primary-secondary, and secondary-secondary products:

```text
theta(q, mu) = [
  1,
  eta(mu),
  q,
  eta(mu) q,
  q_quad,
  N_ANN(q, mu),
  q kron N_ANN(q, mu),
  compact_quad(N_ANN(q, mu))
].
```

For `r = 20` and `qbar = 123`, this gives:

```text
dim(q_quad) = 20 * 21 / 2 = 210
dim(q kron N) = 20 * 123 = 2460
dim(compact_quad(N)) = 123 * 124 / 2 = 7626
num_features = 10545
```

Train/tune and test:

```bash
python3 -B OpInf/stage2_fit_ann_continuous_tuned_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --model-path OpInf/models/ann_nm_mpod_fullquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/FullQuadratic/Training/r20_q123 \
  --with-full-manifold-quadratic \
  --ridges 1e-2,1e0,1e2,1e4,1e6 \
  --rk4-substeps 1,2

python3 -B OpInf/run_ann_manifold_opinf_suite.py \
  --model-path OpInf/models/ann_nm_mpod_fullquadratic_continuous_tuned_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/FullQuadratic/r20_q123
```

Current rollout errors:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| ANN-NM-MPOD, no quadratic | 8.61e-03 | 8.73e-03 | 8.00e-03 |
| ANN-NM-MPOD, partial quadratic | 1.11e-02 | 1.39e-02 | 5.11e-02 |
| ANN-NM-MPOD, full quadratic | 6.63e-03 | 7.59e-03 | 1.23e-02 |

The full quadratic model is the most consistent with substituting the nonlinear
decoder into a quadratic FOM. It is also much larger and needs a dual ridge
solve because the feature count exceeds the number of training samples.

## Polynomial MPOD Baseline

The direct polynomial MPOD analogue replaces the ANN secondary map by

```text
u(t) = u_ref + V q(t) + Vbar Xi g(q(t)),
g(q) = [q^2, ..., q^p] elementwise.
```

For the quadratic-PDE dynamics we use

```text
theta(q, mu) = [1, eta(mu), q, eta(mu) q, q_quad, g(q)].
```

The equivalent-size choice relative to the ANN secondary map is approximately
`p = 7`: with `r = 20`, `dim(g) = 20 * (p - 1) = 120`, close to the ANN's 123
secondary coordinates. In practice, this high-order polynomial map is
numerically fragile in rollout, so the best stable polynomial MPOD found so far
is the smaller `p = 3` model.

Train and test the stable `p=3` case:

```bash
python3 -B OpInf/stage1_fit_manifold_opinf.py \
  --num-primary 20 \
  --num-secondary 123 \
  --polynomial-order 3 \
  --model-path OpInf/models/mpod_quadratic_g_continuous_r20_q123_p3_ridge1e2.npz \
  --results-dir Results/OpInf-MPOD/QuadraticPlusG/Training/r20_q123_p3_ridge1e2 \
  --dynamics-ridge 1e2

python3 -B OpInf/run_manifold_opinf_suite.py \
  --model-path OpInf/models/mpod_quadratic_g_continuous_r20_q123_p3_ridge1e2.npz \
  --results-dir Results/OpInf-MPOD/QuadraticPlusG/r20_q123_p3_ridge1e2
```

Current rollout errors:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| standard continuous OpInf, r=20 | 2.78e-01 | 2.73e-01 | 2.42e-01 |
| polynomial MPOD, p=3, quadratic + g(q) | 6.63e-02 | 5.09e-02 | 5.59e-02 |
| polynomial MPOD, p=7, equivalent-size, ridge=1e2 | unstable | 7.29e-01 | 1.72e-01 |
| ANN-NM-MPOD, no quadratic | 8.61e-03 | 8.73e-03 | 8.00e-03 |
| ANN-NM-MPOD, full quadratic | 6.63e-03 | 7.59e-03 | 1.23e-02 |

This is a useful diagnostic: a high-order polynomial map can match the ANN's
feature count, but not its numerical behavior on this moving-front Burgers
problem. The ANN nonlinear map is not just a cosmetic replacement for the
polynomial map.

## Case 2: Discrete ANN-NM-MPOD

This is the practical fixed-time-step alternative. It avoids derivative
estimation and ODE integration by learning the time-step update directly:

```text
u_k = u_ref + V q_k + Vbar N_ANN(q_k, mu)
q_{k+1} = q_k + W theta(q_k, mu)
theta(q_k, mu) = [1, eta(mu), q_k, eta(mu) q_k, N_ANN(q_k, mu)]
```

There is no explicit quadratic state term `q_k kron q_k`.

Train the discrete operator:

```bash
python3 -B OpInf/stage2_fit_ann_discrete_opinf.py \
  --ann-model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --model-path OpInf/models/ann_manifold_discrete_delta_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Discrete/Training/r20_q123 \
  --ridges 1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2
```

Run the three standard test points:

```bash
python3 -B OpInf/run_ann_discrete_opinf_suite.py \
  --model-path OpInf/models/ann_manifold_discrete_delta_r20_q123.npz \
  --results-dir Results/OpInf-ANN-NM-MPOD/Discrete/r20_q123
```

Run one point manually:

```bash
python3 -B OpInf/run_ann_discrete_opinf.py \
  --model-path OpInf/models/ann_manifold_discrete_delta_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Discrete/one_case \
  --mu1 4.56 \
  --mu2 0.019
```

For the current run, the selected ridge was `1e-2`, and the standard rollout
errors were approximately:

| mu1 | mu2 | relative state error |
| --- | --- | --- |
| 4.56 | 0.019 | 6.42e-03 |
| 4.75 | 0.020 | 5.84e-03 |
| 5.19 | 0.026 | 5.16e-03 |

Outputs are written under:

- `OpInf/models/ann_manifold_discrete_delta_r20_q123.npz`
- `Results/OpInf-ANN-Discrete/Training/r20_q123/`
- `Results/OpInf-ANN-NM-MPOD/Discrete/r20_q123/`

Important caveat: the discrete operator learns the map for the training time
step, currently `dt = 0.05`. It is useful if the online runs keep the same time
step. If the time step changes, retrain or prefer the continuous case.

## Oracle Diagnostic

The oracle suite is not an online ROM. It uses HDM-projected quantities to
separate decoder error from dynamics error:

```bash
python3 -B OpInf/run_ann_manifold_oracle_suite.py \
  --model-path OpInf/models/ann_manifold_linear_plus_ann_r20_q123.npz \
  --results-dir Results/OpInf-ANN-Manifold/Oracle/r20_q123
```

The main reported quantities are:

- `pod_floor_state_error`: best `20 + 123` POD reconstruction error.
- `ann_decoder_on_true_q_state_error`: decoder error using true primary `q`.
- `opinf_rollout_state_error`: normal predictive ROM rollout error.
- `true_qbar_rollout_state_error`: rollout using true secondary coordinates in
  the RHS closure.
- `rhs_ann_on_true_q_relative_error`: learned RHS error on the true trajectory.

Use this diagnostic to decide whether the bottleneck is the ANN decoder, the
inferred dynamics, or phase/time integration.

## Older Baselines

These scripts are kept for comparison, but they are not the current recommended
no-quadratic ANN-manifold workflow.

Linear discrete-time OpInf:

```text
q_{k+1} = c(mu) + A(mu) q_k
```

```bash
python3 -B OpInf/stage1_fit_linear_opinf.py --num-modes 20 --ridge 1e-8
python3 -B OpInf/run_opinf_suite.py
```

Direct quadratic discrete-time OpInf:

```text
q_{k+1} = c(mu) + A(mu) q_k + H(q_k kron q_k)
```

```bash
python3 -B OpInf/stage1_fit_quadratic_opinf.py
python3 -B OpInf/run_quadratic_opinf_suite.py
```

Older polynomial nonlinear-manifold continuous-time OpInf with explicit
quadratic dynamics:

```text
u ~= u_ref + V q + W Xi g(q)
dq/dt = c(mu) + A(mu) q + H(q kron q)
```

```bash
python3 -B OpInf/stage1_fit_manifold_opinf.py
python3 -B OpInf/run_manifold_opinf_suite.py
```

Full-center RBF manifold OpInf without the explicit quadratic term:

```text
u = u_ref + V q + Vbar N_RBF(q, mu)
dq/dt = c(mu) + A(mu) q + B N_RBF(q, mu)
```

```bash
python3 -B OpInf/stage1_fit_rbf_manifold_opinf.py
python3 -B OpInf/run_rbf_manifold_opinf_suite.py
```

To add the explicit quadratic term to the RBF comparison only:

```bash
python3 -B OpInf/stage1_fit_rbf_manifold_opinf.py --with-quadratic
python3 -B OpInf/run_rbf_manifold_opinf_suite.py
```
