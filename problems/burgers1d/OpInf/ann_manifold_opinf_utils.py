"""Utilities for ANN nonlinear-manifold OpInf ROMs."""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from manifold_opinf_utils import continuous_feature_matrix, continuous_feature_vector
from opinf_utils import FEATURE_MODE


ANN_MANIFOLD_MODEL_FAMILY = "ann_manifold_continuous"
ANN_MANIFOLD_DISCRETE_MODEL_FAMILY = "ann_manifold_discrete"


def _as_2d_columns(q_columns):
    q_columns = np.asarray(q_columns, dtype=np.float64)
    if q_columns.ndim == 1:
        q_columns = q_columns.reshape(-1, 1)
    if q_columns.ndim != 2:
        raise ValueError(f"Expected 2D column array, got shape {q_columns.shape}.")
    return q_columns


def build_ann_input_matrix(q_columns, mu, include_mu=True):
    """Build ANN inputs as rows from primary coordinates and optional parameters."""
    q_columns = _as_2d_columns(q_columns)
    x = q_columns.T
    if include_mu:
        mu = np.asarray(mu, dtype=np.float64).reshape(1, -1)
        if mu.shape[1] != 2:
            raise ValueError(f"Expected two parameters, got shape {mu.shape}.")
        x = np.hstack((x, np.repeat(mu, x.shape[0], axis=0)))
    return x


def relative_fro_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.linalg.norm(y_true)
    return float(np.linalg.norm(y_true - y_pred) / (denom if denom > 0.0 else 1.0))


def split_train_validation_indices(n_samples, validation_fraction=0.1, random_seed=42):
    n_samples = int(n_samples)
    if n_samples < 2:
        raise RuntimeError("Need at least two samples for train/validation split.")
    validation_fraction = float(validation_fraction)
    n_val = int(np.floor(validation_fraction * n_samples)) if validation_fraction > 0.0 else 1
    n_val = min(max(1, n_val), n_samples - 1)
    rng = np.random.default_rng(int(random_seed))
    perm = rng.permutation(n_samples)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.maximum(np.asarray(std, dtype=np.float32), float(eps))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.maximum(np.asarray(std, dtype=np.float32), float(eps))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(32, 64, 128, 256, 256)):
        super().__init__()
        dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PODANNModel(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims=(32, 64, 128, 256, 256)):
        super().__init__()
        x_mean = np.asarray(x_mean, dtype=np.float32).reshape(-1)
        x_std = np.asarray(x_std, dtype=np.float32).reshape(-1)
        y_mean = np.asarray(y_mean, dtype=np.float32).reshape(-1)
        y_std = np.asarray(y_std, dtype=np.float32).reshape(-1)
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(x_mean.size, y_mean.size, hidden_dims=hidden_dims)
        self.unscaler = Unscaler(y_mean[None, :], y_std[None, :])

    def forward(self, x_raw):
        x_norm = self.scaler(x_raw)
        y_norm = self.core(x_norm)
        return self.unscaler(y_norm)


def parse_hidden_dims(text):
    if text is None:
        return (32, 64, 128, 256, 256)
    return tuple(int(item.strip()) for item in str(text).split(",") if item.strip())


def train_ann_secondary_map(
    x,
    y,
    hidden_dims=(32, 64, 128, 256, 256),
    validation_fraction=0.1,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-6,
    epochs=2000,
    patience=120,
    min_improve=1e-12,
    clip_grad=1.0,
    random_seed=42,
    device=None,
    print_every=25,
):
    """Train q_secondary = N_ANN(q_primary, mu) with the 2D POD-ANN MLP pattern."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: x {x.shape}, y {y.shape}.")

    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_idx, val_idx = split_train_validation_indices(
        x.shape[0],
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    x_mean = x_train.mean(axis=0)
    x_std = np.maximum(x_train.std(axis=0), 1e-12)
    y_mean = y_train.mean(axis=0)
    y_std = np.maximum(y_train.std(axis=0), 1e-12)

    model = PODANNModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    train_history = []
    val_history = []
    epochs_trained = 0
    print_every = max(1, int(print_every))

    for epoch in range(1, int(epochs) + 1):
        epochs_trained = epoch
        model.train()
        train_loss_acc = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()
            train_loss_acc += float(loss.detach().cpu().item()) * xb.shape[0]

        train_mse = train_loss_acc / x_train.shape[0]
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(x_val_t), y_val_t).detach().cpu().item())
        train_history.append(train_mse)
        val_history.append(val_mse)

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"[ANN-OpInf][EPOCH {epoch:5d}] "
                f"train_mse={train_mse:.6e}, val_mse={val_mse:.6e}, bad={bad_epochs}",
                flush=True,
            )

        if val_mse < best_val - float(min_improve):
            best_val = val_mse
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                print(f"[ANN-OpInf][EarlyStop] epoch={epoch}, best_val={best_val:.6e}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_train_pred = model(torch.from_numpy(x_train).to(device)).cpu().numpy()
        y_val_pred = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
        y_all_pred = model(torch.from_numpy(x).to(device)).cpu().numpy()

    return {
        "torch_model": model,
        "state_dict": {key: val.detach().cpu().clone() for key, val in model.state_dict().items()},
        "device": str(device),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "y_mean": y_mean.astype(np.float64),
        "y_std": y_std.astype(np.float64),
        "train_indices": train_idx.astype(np.int64),
        "validation_indices": val_idx.astype(np.int64),
        "train_history": np.asarray(train_history, dtype=np.float64),
        "val_history": np.asarray(val_history, dtype=np.float64),
        "best_val_mse": float(best_val),
        "epochs_trained": int(epochs_trained),
        "train_relative_error": relative_fro_error(y_train, y_train_pred),
        "validation_relative_error": relative_fro_error(y_val, y_val_pred),
        "full_relative_error": relative_fro_error(y, y_all_pred),
    }


def _ann_model_from_fit(fit):
    return {
        "ann_torch_model": fit["torch_model"],
        "ann_device": fit["device"],
        "ann_include_mu": True,
    }


def predict_ann_secondary(q_primary, mu, model):
    q_primary = _as_2d_columns(q_primary)
    x = build_ann_input_matrix(q_primary, mu, include_mu=bool(model["ann_include_mu"]))
    torch_model = model["ann_torch_model"]
    device = torch.device(model.get("ann_device", "cpu"))
    torch_model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        pred = torch_model(x_t).detach().cpu().numpy().astype(np.float64)
    return pred.T


def ann_manifold_decode(q_primary, basis_primary, basis_secondary, model, u_ref, mu):
    q_primary = _as_2d_columns(q_primary)
    q_secondary = predict_ann_secondary(q_primary, mu, model)
    return u_ref[:, None] + basis_primary @ q_primary + basis_secondary @ q_secondary


def cross_product_matrix(left_columns, right_columns):
    """Return all samplewise products left_i * right_j as rows."""
    left_columns = _as_2d_columns(left_columns)
    right_columns = _as_2d_columns(right_columns)
    if left_columns.shape[1] != right_columns.shape[1]:
        raise ValueError(
            f"Sample mismatch: left {left_columns.shape}, right {right_columns.shape}."
        )
    left = left_columns.T
    right = right_columns.T
    return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)


def compact_quadratic_matrix(q_columns):
    """Return compact q_i q_j terms with i <= j as rows."""
    q_columns = _as_2d_columns(q_columns)
    n_modes, n_samples = q_columns.shape
    out = np.empty((n_samples, n_modes * (n_modes + 1) // 2), dtype=np.float64)
    k = 0
    for i in range(n_modes):
        qi = q_columns[i, :]
        for j in range(i, n_modes):
            out[:, k] = qi * q_columns[j, :]
            k += 1
    return out


def ann_continuous_feature_matrix(
    q_columns,
    mu,
    model,
    include_ann_dynamics=False,
    include_param_ann_dynamics=False,
    include_full_manifold_quadratic=None,
):
    q_columns = _as_2d_columns(q_columns)
    base = continuous_feature_matrix(
        q_columns,
        mu,
        feature_mode=model["feature_mode"],
        include_param_linear=bool(model["include_param_linear"]),
        include_quadratic=bool(model["include_quadratic"]),
        include_higher=bool(model["include_higher"]),
        max_degree=int(model["max_degree"]),
    )
    blocks = [base]
    if include_ann_dynamics:
        if include_full_manifold_quadratic is None:
            include_full_manifold_quadratic = bool(model.get("include_full_manifold_quadratic", False))
        ann = predict_ann_secondary(q_columns, mu, model).T
        blocks.append(ann)
        if include_param_ann_dynamics:
            mu = np.asarray(mu, dtype=np.float64).reshape(-1)
            for val in mu:
                blocks.append(float(val) * ann)
        if include_full_manifold_quadratic:
            ann_columns = ann.T
            blocks.append(cross_product_matrix(q_columns, ann_columns))
            blocks.append(compact_quadratic_matrix(ann_columns))
    return np.hstack(blocks)


def ann_continuous_feature_vector(q, mu, model):
    return ann_continuous_feature_matrix(
        np.asarray(q, dtype=np.float64).reshape(-1, 1),
        mu,
        model,
        include_ann_dynamics=bool(model["include_ann_dynamics"]),
        include_param_ann_dynamics=bool(model["include_param_ann_dynamics"]),
        include_full_manifold_quadratic=bool(model.get("include_full_manifold_quadratic", False)),
    )[0]


def rhs_ann_continuous(q, mu, model):
    if bool(model["include_ann_dynamics"]):
        theta = ann_continuous_feature_vector(q, mu, model)
    else:
        theta = continuous_feature_vector(
            q,
            mu,
            feature_mode=model["feature_mode"],
            include_param_linear=bool(model["include_param_linear"]),
            include_quadratic=bool(model["include_quadratic"]),
            include_higher=bool(model["include_higher"]),
            max_degree=int(model["max_degree"]),
        )
    theta_scaled = (theta - model["x_mean"]) / model["x_scale"]
    return model["operator"] @ theta_scaled


def rollout_ann_continuous_rk4(q0, mu, dt, num_steps, model, max_norm=1e12, substeps=1):
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
            k1 = rhs_ann_continuous(q, mu, model)
            k2 = rhs_ann_continuous(q + 0.5 * h * k1, mu, model)
            k3 = rhs_ann_continuous(q + 0.5 * h * k2, mu, model)
            k4 = rhs_ann_continuous(q + h * k3, mu, model)
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


def save_ann_manifold_model(
    model_path,
    ann_state_dict=None,
    model_family=ANN_MANIFOLD_MODEL_FAMILY,
    **kwargs,
):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    arrays = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            arrays[key] = np.asarray(value)
        elif isinstance(value, (bool, np.bool_)):
            arrays[key] = np.asarray(bool(value), dtype=np.int64)
        elif isinstance(value, (int, np.integer)):
            arrays[key] = np.asarray(int(value), dtype=np.int64)
        elif isinstance(value, (float, np.floating)):
            arrays[key] = np.asarray(float(value), dtype=np.float64)
        else:
            arrays[key] = np.asarray(value)
    if ann_state_dict is not None:
        keys = list(ann_state_dict.keys())
        arrays["ann_state_keys"] = np.asarray(keys)
        for i, key in enumerate(keys):
            value = ann_state_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            arrays[f"ann_state_{i:04d}"] = np.asarray(value)
    arrays["model_family"] = np.asarray(str(model_family))
    np.savez(model_path, **arrays)


def load_ann_manifold_model(model_path, device=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ANN manifold OpInf model not found at {model_path}. "
            "Run OpInf/stage1_fit_ann_manifold_opinf.py first."
        )
    data = np.load(model_path, allow_pickle=False)
    model = {key: data[key] for key in data.files}
    family = str(np.asarray(model["model_family"]).item())
    allowed_families = (ANN_MANIFOLD_MODEL_FAMILY, ANN_MANIFOLD_DISCRETE_MODEL_FAMILY)
    if family not in allowed_families:
        raise ValueError(f"Unsupported model_family={family!r}; expected one of {allowed_families!r}.")
    model["model_family"] = family

    for key in (
        "feature_mode",
        "pod_basis_path",
        "dynamics_feature_type",
        "discrete_time_map",
        "source_ann_model_path",
    ):
        if key in model:
            model[key] = str(np.asarray(model[key]).item())
    for key in ("num_primary", "num_secondary", "num_steps", "max_degree", "num_features"):
        if key in model:
            model[key] = int(np.asarray(model[key]).item())
    for key in (
        "dt",
        "dynamics_ridge",
        "relative_manifold_training_error",
        "relative_derivative_training_error",
        "relative_one_step_delta_training_error",
        "relative_one_step_q_training_error",
        "validation_rollout_error",
        "training_rollout_error",
        "energy_captured_primary",
        "energy_captured_total_basis",
        "ann_validation_fraction",
        "ann_learning_rate",
        "ann_weight_decay",
        "ann_min_improve",
        "ann_clip_grad",
        "ann_best_val_mse",
        "ann_train_relative_error",
        "ann_validation_relative_error",
        "ann_full_relative_error",
    ):
        if key in model:
            model[key] = float(np.asarray(model[key]).item())
    for key in (
        "ann_batch_size",
        "ann_epochs_requested",
        "ann_epochs_trained",
        "ann_patience",
        "ann_random_seed",
        "num_validation_trajectories",
        "num_training_trajectories",
        "rk4_substeps",
    ):
        if key in model:
            model[key] = int(np.asarray(model[key]).item())
    for key in (
        "include_param_linear",
        "include_quadratic",
        "include_higher",
        "ann_include_mu",
        "include_ann_dynamics",
        "include_param_ann_dynamics",
        "include_full_manifold_quadratic",
    ):
        if key in model:
            model[key] = bool(int(np.asarray(model[key]).item()))
    for key in (
        "operator",
        "x_mean",
        "x_scale",
        "ann_x_mean",
        "ann_x_std",
        "ann_y_mean",
        "ann_y_std",
    ):
        if key in model:
            model[key] = np.asarray(model[key], dtype=np.float64)

    hidden_dims = tuple(int(v) for v in np.asarray(model["ann_hidden_dims"]).reshape(-1))
    model["ann_hidden_dims"] = hidden_dims
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch_model = PODANNModel(
        x_mean=model["ann_x_mean"],
        x_std=model["ann_x_std"],
        y_mean=model["ann_y_mean"],
        y_std=model["ann_y_std"],
        hidden_dims=hidden_dims,
    ).to(device)
    state_keys = [str(key) for key in np.asarray(model["ann_state_keys"]).reshape(-1)]
    state_dict = {}
    for i, key in enumerate(state_keys):
        state_dict[key] = torch.as_tensor(np.asarray(model[f"ann_state_{i:04d}"]))
    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.eval()
    model["ann_torch_model"] = torch_model
    model["ann_device"] = str(device)
    return model


def predict_next_q_ann_discrete(q, mu, model):
    theta = ann_continuous_feature_vector(q, mu, model)
    theta_scaled = (theta - model["x_mean"]) / model["x_scale"]
    update = np.asarray(model["operator"], dtype=np.float64) @ theta_scaled
    mode = str(model.get("discrete_time_map", "delta"))
    if mode == "delta":
        return np.asarray(q, dtype=np.float64).reshape(-1) + update
    if mode == "next":
        return update
    raise ValueError(f"Unsupported discrete_time_map={mode!r}.")


def rollout_ann_discrete(q0, mu, num_steps, model, max_norm=1e12):
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q_snaps = np.zeros((q0.size, int(num_steps) + 1), dtype=np.float64)
    q_snaps[:, 0] = q0
    q = q0.copy()
    stable_steps = int(num_steps)
    unstable_reason = ""
    for istep in range(int(num_steps)):
        q = predict_next_q_ann_discrete(q, mu, model)
        if (not np.all(np.isfinite(q))) or np.linalg.norm(q) > float(max_norm):
            stable_steps = istep
            unstable_reason = f"unstable at step {istep + 1}"
            q_snaps[:, istep + 1 :] = np.nan
            break
        q_snaps[:, istep + 1] = q
    return q_snaps, stable_steps, unstable_reason


def model_stub_from_ann_fit(
    fit,
    feature_mode=FEATURE_MODE,
    include_param_linear=True,
    include_quadratic=False,
    include_higher=False,
    max_degree=2,
    ann_include_mu=True,
    include_ann_dynamics=True,
    include_param_ann_dynamics=False,
    include_full_manifold_quadratic=False,
):
    model = _ann_model_from_fit(fit)
    model.update(
        {
            "feature_mode": feature_mode,
            "include_param_linear": bool(include_param_linear),
            "include_quadratic": bool(include_quadratic),
            "include_higher": bool(include_higher),
            "max_degree": int(max_degree),
            "ann_include_mu": bool(ann_include_mu),
            "include_ann_dynamics": bool(include_ann_dynamics),
            "include_param_ann_dynamics": bool(include_param_ann_dynamics),
            "include_full_manifold_quadratic": bool(include_full_manifold_quadratic),
        }
    )
    return model
