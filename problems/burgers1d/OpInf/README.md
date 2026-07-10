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

`stage1_fit_manifold_opinf.py` fits this with a single fixed
`--dynamics-ridge` (fast, untuned). Prefer
`stage2_fit_manifold_continuous_tuned_opinf.py`, which grid-searches ridge and
RK4 substeps with held-out-trajectory validation, the same protocol
Case 0/2 use below -- MPOD is not a fair baseline for the AL/PQ/QC comparison
unless it is tuned the same way. This library is also numerically fragile
at low ridge (`RuntimeWarning: overflow` during the sweep is expected, not a
bug); the script's full-data-refit stability check exists to keep the search
from quietly selecting an unstable "best" candidate that only fails once
refit on all 9 trajectories.

Commands:

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

## Case 2: ANN-NM-MPOD (AL / PQ / QC)

Decoder (shared across all three dynamics libraries below -- train once):

```text
u(t) = u_ref + V q(t) + Vbar N_ANN(q(t), mu)
```

`--operator-mode {al,pq,qc}` selects which of the report's three dynamics
libraries is regressed on top of that decoder:

```text
al: dq/dt = W [1, eta(mu), q, eta(mu) kron q, z]
pq: dq/dt = W [1, eta(mu), q, eta(mu) kron q, q_quad, z]
qc: dq/dt = W [1, eta(mu), q, eta(mu) kron q, q_quad, z, q kron z, z_quad]
    z = N_ANN(q, mu)
```

For `r=10`, `rbar=133`, and five parameter features: al has `199` RHS
features, pq has `254`, and qc has `10495`:

```text
standard block = 1 + 5 + 10 + 5*10 + 10*11/2 = 121
z block        = 133
q kron z       = 10*133 = 1330   (qc only)
z_quad         = 133*134/2 = 8911 (qc only)
al   = 121 + 133 - 55            = 199   (al drops q_quad)
pq   = 121 + 133                 = 254
qc   = 121 + 133 + 1330 + 8911   = 10495
```

qc's per-call cost is not negligible -- see the report's online-cost table:
its `q kron z`/`z_quad` construction and the extra RK4 substeps it needs for
stability make it roughly 70x slower online than the $N=5000$ full-order
model, despite being the most accurate. `al` is within a factor of ~2 of
qc's accuracy at close to standard-OpInf's cost and is the practical default.

Commands (`qc` shown; swap `--operator-mode` and the paths for `al`/`pq`):

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

The ANN decoder's validation split holds out whole trajectories, not
scattered snapshots (see `stage1_fit_ann_manifold_opinf.py`'s `group_ids`
argument to `train_ann_secondary_map`); a plain random row split leaks
information across time-adjacent snapshots and gives an over-optimistic
early-stopping curve.

## Current Results

Relative state errors over all finite snapshots, all rows tuned by the same
ridge x RK4-substep + held-out-trajectory validation protocol:

| model | 4.56, 0.019 | 4.75, 0.020 | 5.19, 0.026 |
| --- | ---: | ---: | ---: |
| Standard quadratic OpInf, `r=10` | 9.11e-02 | 9.11e-02 | 8.74e-02 |
| Induced MPOD (tuned), `r=10,rbar=133,p=2` | 8.19e-02 | 8.39e-02 | 8.12e-02 |
| ANN AL-NM-MPOD, `r=10,rbar=133` | 1.14e-02 | 9.01e-03 | 6.52e-03 |
| ANN PQ-NM-MPOD, `r=10,rbar=133` | 1.37e-02 | 1.33e-02 | 1.32e-02 |
| ANN QC-NM-MPOD, `r=10,rbar=133` | 5.85e-03 | 5.79e-03 | 4.96e-03 |

The retained suite is the theory-consistent continuous comparison used in the
current report for a quadratic Burgers FOM. See
`../../reports/nm_mpod_proposal/` for the extrapolation test, online-cost
table, and the induced-MPOD $p=5$ instability finding (higher polynomial
order is not monotonically better once tuned fairly).
