import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from .patchtst.patchtst import PatchTSTModel
from .config import PatchTSTConfig
from .utils import class_to_dict

class PatchTSTLightningModule(L.LightningModule):
    # def __init__(self, 
    #              enc_in = 1,
    #              seq_len = 100,
    #              pred_len = None,
    #              e_layers = 3,
    #              n_heads = 8,
    #              d_model = 128,
    #              d_ff = 256,
    #              dropout = 0.1,
    #              fc_dropout = 0.1,
    #              head_dropout = 0.0,
    #              individual = False,
    #              patch_len = 16,
    #              stride = 8,
    #              padding_patch = None,
    #              revin = True,
    #              affine = True,
    #              subtract_last = False,
    #              decomposition = False,
    #              kernel_size = 25,
    #              loss = "mse",
    #              target_dim = None
    #              ):
    def __init__(self, PatchTSTConfig_cls, ModelConfig_cls, TrainConfig_cls):
        super().__init__()
        self.patchtst_cfg = class_to_dict(PatchTSTConfig_cls)
        self.model_cfg = class_to_dict(ModelConfig_cls)
        self.train_cfg = class_to_dict(TrainConfig_cls)

        # Loss configuration
        self.loss_name = self.train_cfg.get("loss", "mse").lower()
        self.loss_fn = nn.MSELoss() if self.loss_name in {"mse", "l2"} else nn.L1Loss()

        # Core PatchTST model
        self.model = PatchTSTModel(self.patchtst_cfg)
        self.target_dim = self.train_cfg.get("target_dim", None)
        self.out_proj = nn.Linear(self.patchtst_cfg.get("enc_in", 1), int(self.target_dim)) if self.target_dim is not None else None

    @staticmethod
    def _flatten_channels(x):
        if x.ndim == 2:  # [B, L]
            return x.unsqueeze(-1)
        if x.ndim == 3:  # [B, L, C]
            return x
        if x.ndim >= 4:
            b, l = x.shape[0], x.shape[1]
            return x.reshape(b, l, -1)
        raise ValueError(f"Unexpected input ndim: {x.ndim}")

    def _ensure_proj(self, y):
        """Create output projection lazily if target_dim wasn't provided."""
        if self.out_proj is not None:
            return
        # infer target dimension
        y_flat = self._flatten_channels(y)
        self.target_dim = y_flat.shape[-1]
        self.out_proj = nn.Linear(self.patchtst_cfg.get("enc_in", 1), int(self.target_dim)).to(self.device)

    def forward(self, x):
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
        if self.patchtst_cfg.get("enc_in", 1) == 1 and x_flat.shape[-1] != 1 and (not hasattr(self, "_enc_in_inferred")):
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
        lr = self.train_cfg.get("lr", 1e-3)
        weight_decay = self.train_cfg.get("weight_decay", 1e-2)
        warmup_ratio = self.train_cfg.get("warmup_ratio", 0.05)
        pretrain_epochs = self.train_cfg.get("pretrain_epochs", getattr(self.trainer, "max_epochs", 1))

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
        if self.logger is not None:
            self.logger.log(f"Epoch {trainer.current_epoch + 1} duration: {duration:.2f} seconds", log_type="log")
