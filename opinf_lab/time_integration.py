"""Time integration helpers for inferred reduced ODEs."""

import numpy as np


def rollout_rk4(rhs, q0, dt, num_steps, max_norm=1e12, substeps=1):
    """Roll out ``dq/dt = rhs(q)`` using fixed-step RK4."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q_snaps = np.zeros((q0.size, int(num_steps) + 1), dtype=np.float64)
    q_snaps[:, 0] = q0
    q = q0.copy()
    stable_steps = int(num_steps)
    unstable_reason = ""
    substeps = max(1, int(substeps))
    h = float(dt) / float(substeps)

    for istep in range(int(num_steps)):
        for _ in range(substeps):
            k1 = np.asarray(rhs(q), dtype=np.float64)
            k2 = np.asarray(rhs(q + 0.5 * h * k1), dtype=np.float64)
            k3 = np.asarray(rhs(q + 0.5 * h * k2), dtype=np.float64)
            k4 = np.asarray(rhs(q + h * k3), dtype=np.float64)
            q = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if (not np.all(np.isfinite(q))) or np.linalg.norm(q) > float(max_norm):
                stable_steps = istep
                unstable_reason = f"unstable at step {istep + 1}"
                q_snaps[:, istep + 1 :] = np.nan
                break
        if unstable_reason:
            break
        q_snaps[:, istep + 1] = q
    return q_snaps, stable_steps, unstable_reason
