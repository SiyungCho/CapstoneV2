# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dataclasses import dataclass
# import lightning as L
# from torch.optim.lr_scheduler import LambdaLR

# from patchtst import PatchTSTModel

# class PatchTSTLightningModule(L.LightningModule):
#     def __init__(self, hparams):
#         super().__init__()
#         self.save_hyperparameters(hparams)
#         self.model = PatchTSTModel(hparams)

#     def _common_step(self, batch, step_type):
       
#         return 

#     def training_step(self, batch, batch_idx): return self._common_step(batch, 'train')
#     def validation_step(self, batch, batch_idx): return self._common_step(batch, 'val')

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         try:
#             total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
#         except Exception:
#             total_steps = self.hparams.pretrain_epochs * 250
#         warmup_steps = int(total_steps * self.hparams.warmup_ratio)

#         def lr_lambda(current_step: int):
#             if current_step < warmup_steps:
#                 return float(current_step) / float(max(1, warmup_steps))
#             return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

#         scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda), 'interval': 'step', 'frequency': 1}
#         return [optimizer], [scheduler]

# class EpochTimer(L.Callback):
#     """Callback to time each training epoch and log the duration."""
#     def __init__(self, logger):
#         super().__init__()
#         self.logger = logger
#         self.epoch_times = []
#         self.start_time = 0

#     def on_train_epoch_start(self, trainer, pl_module):
#         """Record the start time at the beginning of each training epoch."""
#         self.start_time = time.time()

#     def on_train_epoch_end(self, trainer, pl_module):
#         """Calculate and log the epoch duration at the end of each training epoch."""
#         end_time = time.time()
#         duration = end_time - self.start_time
        
#         self.epoch_times.append(duration)
#         self.logger.log(f"Epoch {trainer.current_epoch + 1} duration: {duration:.2f} seconds", log_type="log")

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from .patchtst import PatchTSTModel


@dataclass
class PatchTSTConfig:
    """Configuration holder expected by patchtst.Model.

    The upstream PatchTST code expects `configs.<field>` style access.
    """
    # data / io
    enc_in: int = 1          # number of input channels (variables)
    seq_len: int = 100       # context window length
    pred_len: int = 100      # prediction window length (output sequence length)

    # transformer
    e_layers: int = 3
    n_heads: int = 8
    d_model: int = 128
    d_ff: int = 256
    dropout: float = 0.1
    fc_dropout: float = 0.1
    head_dropout: float = 0.0

    # head
    individual: bool = False

    # patching
    patch_len: int = 16
    stride: int = 8
    padding_patch: Optional[str] = None  # 'end' or None

    # RevIN
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False

    # decomposition
    decomposition: bool = False
    kernel_size: int = 25


def _to_namespace(hparams: Union[Dict[str, Any], Any]) -> SimpleNamespace:
    """Convert dict/namespace/dataclass to a SimpleNamespace for attribute access."""
    if isinstance(hparams, SimpleNamespace):
        return hparams
    if isinstance(hparams, dict):
        return SimpleNamespace(**hparams)
    # dataclass or argparse Namespace or any object with __dict__
    if hasattr(hparams, "__dict__"):
        return SimpleNamespace(**vars(hparams))
    raise TypeError(f"Unsupported hparams type: {type(hparams)}")


class PatchTSTLightningModule(L.LightningModule):
    """Lightning module for training PatchTST on (x -> y) sequences.

    Expected batch shapes from the provided dataloader:
      x: [B, L, 40, 3] (or [B, L, C])
      y: [B, L, Dy]    (or [B, L])

    We flatten x's trailing dimensions into a channel dimension, run PatchTST,
    then (optionally) project PatchTST's output channels to the target dimension.
    """

    def __init__(self, hparams: Union[Dict[str, Any], Any]):
        super().__init__()
        hparams_ns = _to_namespace(hparams)
        self.save_hyperparameters(vars(hparams_ns))

        # Loss configuration
        self.loss_name = getattr(hparams_ns, "loss", "mse").lower()
        self.loss_fn = nn.MSELoss() if self.loss_name in {"mse", "l2"} else nn.L1Loss()

        # Build model config (PatchTST expects attribute access)
        cfg = PatchTSTConfig(
            enc_in=getattr(hparams_ns, "enc_in", 1),
            seq_len=getattr(hparams_ns, "seq_len", 100),
            pred_len=getattr(hparams_ns, "pred_len", getattr(hparams_ns, "seq_len", 100)),
            e_layers=getattr(hparams_ns, "e_layers", 3),
            n_heads=getattr(hparams_ns, "n_heads", 8),
            d_model=getattr(hparams_ns, "d_model", 128),
            d_ff=getattr(hparams_ns, "d_ff", 256),
            dropout=getattr(hparams_ns, "dropout", 0.1),
            fc_dropout=getattr(hparams_ns, "fc_dropout", 0.1),
            head_dropout=getattr(hparams_ns, "head_dropout", 0.0),
            individual=getattr(hparams_ns, "individual", False),
            patch_len=getattr(hparams_ns, "patch_len", 16),
            stride=getattr(hparams_ns, "stride", 8),
            padding_patch=getattr(hparams_ns, "padding_patch", None),
            revin=getattr(hparams_ns, "revin", True),
            affine=getattr(hparams_ns, "affine", True),
            subtract_last=getattr(hparams_ns, "subtract_last", False),
            decomposition=getattr(hparams_ns, "decomposition", False),
            kernel_size=getattr(hparams_ns, "kernel_size", 25),
        )
        self.cfg = cfg

        # Core PatchTST model
        self.model = PatchTSTModel(cfg)

        # Optional output projection (PatchTST outputs `enc_in` channels)
        # If target_dim is not provided, we infer from the first batch at runtime.
        self.target_dim: Optional[int] = getattr(hparams_ns, "target_dim", None)
        self.out_proj: Optional[nn.Module] = None
        if self.target_dim is not None:
            self.out_proj = nn.Linear(cfg.enc_in, int(self.target_dim))

    @staticmethod
    def _flatten_channels(x: torch.Tensor) -> torch.Tensor:
        """Flatten trailing dims into a single channel dim: [B, L, ...] -> [B, L, C]."""
        if x.ndim == 2:  # [B, L]
            return x.unsqueeze(-1)
        if x.ndim == 3:  # [B, L, C]
            return x
        if x.ndim >= 4:
            b, l = x.shape[0], x.shape[1]
            return x.reshape(b, l, -1)
        raise ValueError(f"Unexpected input ndim: {x.ndim}")

    def _ensure_proj(self, y: torch.Tensor):
        """Create output projection lazily if target_dim wasn't provided."""
        if self.out_proj is not None:
            return
        # infer target dimension
        y_flat = self._flatten_channels(y)
        self.target_dim = y_flat.shape[-1]
        self.out_proj = nn.Linear(self.cfg.enc_in, int(self.target_dim)).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self._flatten_channels(x)
        return self.model(x_flat)

    def _common_step(self, batch, step_type: str):
        # Unpack batch (dataset optionally returns deltas as 3rd item)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Batch must be a tuple/list like (x, y) or (x, y, ...).")

        x = x.float()
        y = y.float()

        # Flatten any extra dims into channels
        x_flat = self._flatten_channels(x)
        y_flat = self._flatten_channels(y)

        # If user didn't set enc_in, infer it from x on first step
        if self.cfg.enc_in == 1 and x_flat.shape[-1] != 1 and (not hasattr(self, "_enc_in_inferred")):
            # Update cfg.enc_in for correctness in logging; PatchTST already built, so warn if mismatch
            self._enc_in_inferred = True  # avoid repeating
            if x_flat.shape[-1] != self.model.model.n_vars if hasattr(self.model, "model") else x_flat.shape[-1]:
                # can't rebuild safely mid-run; but we can at least note the mismatch
                pass

        # Forward
        y_hat = self.model(x_flat)  # [B, pred_len, enc_in]

        # Align lengths if needed
        # Common case with this dataset: y is [B, L, Dy] and pred_len == L
        min_len = min(y_hat.shape[1], y_flat.shape[1])
        y_hat = y_hat[:, :min_len]
        y_flat = y_flat[:, :min_len]

        # Output projection to target dim
        self._ensure_proj(y_flat)
        y_pred = self.out_proj(y_hat)  # [B, L, Dy]

        loss = self.loss_fn(y_pred, y_flat)
        mae = F.l1_loss(y_pred, y_flat)

        # Logging
        self.log(f"{step_type}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log(f"{step_type}/mae", mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.shape[0])

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr", 1e-3)
        weight_decay = getattr(self.hparams, "weight_decay", 1e-2)
        warmup_ratio = getattr(self.hparams, "warmup_ratio", 0.05)
        pretrain_epochs = getattr(self.hparams, "pretrain_epochs", getattr(self.trainer, "max_epochs", 1))

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # total steps (best-effort)
        try:
            total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        except Exception:
            total_steps = int(pretrain_epochs) * 250

        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step: int):
            if warmup_steps <= 0:
                return 1.0
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class EpochTimer(L.Callback):
    """Callback to time each training epoch and log the duration."""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.epoch_times = []
        self.start_time = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        duration = end_time - self.start_time
        self.epoch_times.append(duration)
        # logger is expected to have `.log(msg, log_type=...)` as used in the starter file.
        if self.logger is not None:
            self.logger.log(
                f"Epoch {trainer.current_epoch + 1} duration: {duration:.2f} seconds",
                log_type="log",
            )
