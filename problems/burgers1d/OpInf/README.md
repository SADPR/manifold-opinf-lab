# OpInf

Continuous Operator Inference ROMs for the 1D parametric Burgers benchmark.

Run all commands from:

```bash
cd /home/kratos/manifold-opinf-lab/problems/burgers1d
```

If needed, rebuild the FOM snapshots and POD basis first:

```bash
python3 -B run_fom_training.py
python3 -B POD/stage1_build_pod_basis.py
```

Notation:

- `r` is the number of evolved primary POD coordinates.
- `rbar` is the number of secondary POD coordinates represented algebraically.
- Some filenames still use the historical token `q133`; read this as `rbar=133`, not as the evolved vector `q(t)`.

## Case 0: Standard Continuous OpInf

Decoder:

```text
u(t) = u_ref + V q(t)
```

Reduced dynamics:

```text
dq/dt = W [1, eta(mu), q, eta(mu) kron q, q_quad]
```

Commands:

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

## Case 1: Polynomial MPOD

Decoder:

```text
u(t) = u_ref + V q(t) + Vbar Xi g_p(q(t))
```

Reduced dynamics use the induced higher-monomial library obtained by
substituting the polynomial manifold into a quadratic FOM:

```text
dq/dt = W [1, eta(mu), q, eta(mu) kron q, q_quad, ghat(q)]
```

Commands:

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

## Case 2: ANN-NM-MPOD Full Quadratic

Decoder:

```text
u(t) = u_ref + V q(t) + Vbar N_ANN(q(t), mu)
```

The full state-quadratic reduced dynamics are:

```text
dq/dt = W [
  1,
  eta(mu),
  q,
  eta(mu) kron q,
  q_quad,
  z,
  q kron z,
  z_quad
],  z = N_ANN(q, mu)
```

For `r=10`, `rbar=133`, and five parameter features, this has `10495` RHS
features:

```text
standard block = 1 + 5 + 10 + 5*10 + 10*11/2 = 121
z block        = 133
q kron z       = 10*133 = 1330
z_quad         = 133*134/2 = 8911
total          = 10495
```

Commands:

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

## Current Results

Relative state errors over all finite snapshots:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| Standard quadratic OpInf, `r=10` | 9.11e-02 | 9.11e-02 | 8.74e-02 |
| MPOD, `r=10,rbar=133,p=2` | 8.19e-02 | 8.39e-02 | 8.12e-02 |
| ANN-NM-MPOD full quadratic, `r=10,rbar=133` | 5.97e-03 | 5.76e-03 | 6.56e-03 |

The retained suite is the theory-consistent continuous comparison used in the
current report for a quadratic Burgers FOM.
