import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from patchtst.patchtst import PatchTSTModel
from utils import class_to_dict

class PatchTSTLightningModule(L.LightningModule):
    def __init__(self, PatchTSTConfig_cls, ModelConfig_cls, TrainConfig_cls):
        super().__init__()
        self.patchtst_cfg = class_to_dict(PatchTSTConfig_cls)
        self.model_cfg = class_to_dict(ModelConfig_cls)
        self.train_cfg = class_to_dict(TrainConfig_cls)
        self.save_hyperparameters(self.patchtst_cfg)
        self.save_hyperparameters(self.model_cfg)
        self.save_hyperparameters(self.train_cfg)

        # Loss configuration
        self.loss_name = self.train_cfg.get("loss", "mse").lower()
        self.loss_fn = nn.MSELoss() if self.loss_name in {"mse", "l2"} else nn.L1Loss()

        # Core PatchTST model
        self.model = PatchTSTModel(**self.patchtst_cfg)
        self.target_dim = self.train_cfg.target_dim
        self.out_proj = nn.Linear(self.patchtst_cfg.enc_in, int(self.target_dim))

    @staticmethod
    def _flatten_channels(x):
        if x.ndim == 2:  # [B, L]
            return x.unsqueeze(-1)
        if x.ndim == 3:  # [B, L, C]
            return x
        if x.ndim == 4:  # [B, L, C, D] -> [B, L, C*D]
            B, L, C, D = x.shape
            return x.view(B, L, C * D)
        raise ValueError(f"Unexpected input ndim: {x.ndim}")

    def forward(self, x):
        x_flat = self._flatten_channels(x)
        return self.model(x_flat)

    def _common_step(self, batch, step_type):
        x, y = batch[0], batch[1]
        x = x.float()
        y = y.float()

        x_flat = self._flatten_channels(x)
        y_flat = self._flatten_channels(y)

        y_pred = self.model(x_flat)

        loss = self.loss_fn(y_pred, y_flat)
        mae = F.l1_loss(y_pred, y_flat)

        self.log(f'{step_type}/loss', loss, on_step=(step_type=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{step_type}/mae', mae, on_step=(step_type=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._common_step(batch, "val")

    def configure_optimizers(self):
        lr = self.train_cfg.lr
        weight_decay = self.train_cfg.weight_decay
        warmup_ratio = self.train_cfg.warmup_ratio
        pretrain_epochs = self.train_cfg.pretrain_epochs

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        try:
            total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        except Exception:
            total_steps = int(pretrain_epochs) * 250
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step: int):
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
