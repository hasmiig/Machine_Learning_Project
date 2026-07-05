"""
PyTorch tabular models for Seattle Airbnb price prediction.

All three model classes expose a common fusion interface so a downstream
multi-modal head can concatenate embeddings from each modality before
making a final prediction:

    tab_emb = tabular_model.encode(x_tab)   # [B, D_tab]
    txt_emb = text_model.encode(x_txt)       # [B, D_txt]
    img_emb = image_model.encode(x_img)      # [B, D_img]
    fused   = torch.cat([tab_emb, txt_emb, img_emb], dim=1)
    pred    = fusion_head(fused)

Each model stores its embedding dimensionality in model.embed_dim.
Target: log_price = log(price) — use np.exp() / torch.exp() to recover dollars.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Column group constants ─────────────────────────────────────────────────────

_DROP_COLS: list[str] = ["id", "neighbourhood_group_cleansed"]

_BOOL_STR: list[str] = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

# -1 is a sentinel meaning "no data reported"
_RATE_COLS: list[str] = ["host_response_rate", "host_acceptance_rate"]

_OHE_COLS: list[str] = [
    "host_response_time",
    "host_verifications",
    "room_type",
    "property_type",
    "host_listing_bucket",
    "neighbourhood_cleansed",
]


# ── Feature encoder ────────────────────────────────────────────────────────────

class FeatureEncoder:
    """Stateful encoder; call fit() on training data only to prevent leakage.

    Encoding pipeline (applied in order):
        1. Drop id and neighbourhood_group_cleansed
        2. Bool strings ('t' / 'f') → float 0 / 1
        3. Rate sentinel (-1) → 0.0 + paired <col>_missing binary flag
        4. OneHotEncoder on all string categoricals
        5. StandardScaler across every encoded dimension
    """

    def __init__(self) -> None:
        self._ohe: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None
        # Column lists fixed at fit time; reused by transform for stable ordering.
        self._ohe_present: list[str] = []
        self._num_cols: list[str] = []
        self.n_features: int = 0

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(X_df: pd.DataFrame) -> pd.DataFrame:
        X = X_df.copy()
        X = X.drop(columns=[c for c in _DROP_COLS if c in X.columns])
        for col in _BOOL_STR:
            if col in X.columns:
                X[col] = (X[col] == "t").astype(np.float32)
        for col in _RATE_COLS:
            if col in X.columns:
                # Add the flag before replacing so the equality check is clean.
                X[f"{col}_missing"] = (X[col] == -1.0).astype(np.float32)
                X[col] = X[col].replace(-1.0, 0.0)
        return X

    def _build_array(self, X: pd.DataFrame) -> np.ndarray:
        num = X[self._num_cols].values.astype(np.float32)
        ohe = self._ohe.transform(X[self._ohe_present])
        return np.hstack([num, ohe]).astype(np.float32)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_df: pd.DataFrame) -> "FeatureEncoder":
        X = self._preprocess(X_df)
        self._ohe_present = [c for c in _OHE_COLS if c in X.columns]
        # Fix column order at fit time so transform is stable across DataFrames.
        self._num_cols = [c for c in X.columns if c not in self._ohe_present]

        self._ohe = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self._ohe.fit(X[self._ohe_present])

        full = self._build_array(X)
        self._scaler = StandardScaler()
        self._scaler.fit(full)
        self.n_features = full.shape[1]
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self._preprocess(X_df)
        full = self._build_array(X)
        return self._scaler.transform(full).astype(np.float32)

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.fit(X_df).transform(X_df)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ── Base mixin ─────────────────────────────────────────────────────────────────

class _TabularBase(nn.Module):
    """Shared interface for all tabular models."""

    embed_dim: int  # overridden in each subclass __init__

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate embedding tensor — the hook for fusion."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict_numpy(
        self,
        X: np.ndarray,
        device: torch.device | None = None,
    ) -> np.ndarray:
        self.eval()
        dev = device or next(self.parameters()).device
        x_t = torch.tensor(X, dtype=torch.float32, device=dev)
        return self(x_t).squeeze(-1).cpu().numpy()


# ── LinearTabular ──────────────────────────────────────────────────────────────

class LinearTabular(_TabularBase):
    """Single linear layer from encoded features to scalar log-price.

    encode() returns the scaled input itself; embed_dim == n_features.
    This gives fusion an unprocessed but standardised tabular representation.

    bias_init should be set to y_train.mean() so gradient descent starts from
    the dummy baseline (MSE ≈ var(y) ≈ 0.34) instead of from zero (MSE ≈ 25).
    """

    def __init__(self, n_features: int, bias_init: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = n_features
        self.head = nn.Linear(n_features, 1)
        if bias_init != 0.0:
            nn.init.constant_(self.head.bias, bias_init)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── MLPTabular ─────────────────────────────────────────────────────────────────

class MLPTabular(_TabularBase):
    """Stack of [Linear → BatchNorm1d → GELU → Dropout] blocks.

    encode() returns the output of the last hidden block; embed_dim == hidden_dims[-1].
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_dims[-1]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ── ResNetTabular ──────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Post-norm residual block: output = x + LayerNorm(x + ff(x))."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        expand = 4 * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand, d_model),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ln(x + self.ff(x))


class ResNetTabular(_TabularBase):
    """
    Differentiable analogue of gradient boosting: each residual block refines
    the representation by learning the residual left by all preceding blocks,
    mirroring how boosting adds weak learners to correct the ensemble's current
    errors. True random forests and standard GBDT implementations are not
    differentiable and cannot be fine-tuned end-to-end inside a multi-modal
    fusion pipeline — this architecture fills that role.

    Architecture: Linear(n_features → d_model) → N × _ResBlock → Linear(d_model → 1).
    encode() returns the d_model-dimensional output of the final block.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embed_dim = d_model
        self.proj = nn.Linear(n_features, d_model)
        self.blocks = nn.Sequential(*[_ResBlock(d_model, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(
    model: _TabularBase,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train with AdamW + CosineAnnealingLR; restore best weights on early stop.

    Returns
    -------
    dict with keys: train_losses, val_losses, best_val_rmse, best_epoch
    """
    dev = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(dev)

    train_ds = TabularDataset(X_train, y_train)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_X = torch.tensor(X_val, dtype=torch.float32, device=dev)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = math.inf
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(dev), y_b.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(X_b).squeeze(-1), y_b)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(X_b)
        scheduler.step()

        train_loss = running / len(train_ds)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X).squeeze(-1), val_y).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  (best epoch={best_epoch})")
            break

    model.load_state_dict(best_weights)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_rmse": math.sqrt(best_val_loss),
        "best_epoch": best_epoch,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model: _TabularBase,
    X: np.ndarray,
    y: np.ndarray,
    label: str = "",
    device: torch.device | None = None,
) -> dict:
    """Print and return regression metrics in log-price and dollar space."""
    preds = model.predict_numpy(X, device)
    y = np.asarray(y)

    rmse_log = float(np.sqrt(np.mean((preds - y) ** 2)))
    mae_log  = float(np.mean(np.abs(preds - y)))
    ss_res   = np.sum((y - preds) ** 2)
    ss_tot   = np.sum((y - y.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot)

    p_t = np.exp(y)
    p_p = np.exp(preds)
    rmse_dollar      = float(np.sqrt(np.mean((p_t - p_p) ** 2)))
    mae_dollar       = float(np.mean(np.abs(p_t - p_p)))
    median_ae_dollar = float(np.median(np.abs(p_t - p_p)))
    mape             = float(np.mean(np.abs((p_t - p_p) / p_t)) * 100)

    prefix = f"[{label}]" if label else ""
    print(f"{prefix:<26}  RMSE log={rmse_log:.4f}  MAE log={mae_log:.4f}  R²={r2:.4f}")
    print(f"{'':26}  RMSE $={rmse_dollar:>10,.0f}  MAE $={mae_dollar:>10,.0f}  "
          f"MedAE $={median_ae_dollar:>10,.0f}  MAPE={mape:.1f}%")
    return {
        "rmse_log": rmse_log, "mae_log": mae_log, "r2": r2,
        "rmse_dollar": rmse_dollar, "mae_dollar": mae_dollar,
        "median_ae_dollar": median_ae_dollar, "mape": mape,
    }
