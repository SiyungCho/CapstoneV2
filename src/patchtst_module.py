import time
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from patchtst.patchtst import PatchTSTModel
from utils import class_to_dict

def _merge_patchtst_configs(patchtst_cfg, model_cfg, train_cfg):
    cfg = dict(patchtst_cfg)
    if "patch_stride" in model_cfg and "stride" not in cfg:
        cfg["stride"] = model_cfg["patch_stride"]
    if "patch_stride" in model_cfg:
        cfg["stride"] = model_cfg["patch_stride"]
    if "patch_len" in model_cfg:
        cfg["patch_len"] = model_cfg["patch_len"]

    # Transformer dims
    for k in ("d_model", "d_ff", "e_layers", "n_heads", "dropout", "revin"):
        if k in model_cfg:
            cfg[k] = model_cfg[k]

    cfg["enc_in"] = int(train_cfg["enc_in"])
    cfg["c_out"] = int(train_cfg["target_dim"])

    # Ensure output sequence length matches labels (y is windowed to seq_len)
    # If user didn't explicitly set pred_len, default to seq_len.
    cfg["seq_len"] = int(cfg.get("seq_len", train_cfg.get("seq_len", 100)))
    cfg["pred_len"] = int(cfg.get("pred_len", cfg["seq_len"]))
    # Commonly for seq2seq regression we want pred_len == seq_len.
    cfg["pred_len"] = cfg["seq_len"]

    return SimpleNamespace(**cfg)

class PatchTSTLightningModule(L.LightningModule):
    def __init__(self, PatchTSTConfig_cls, ModelConfig_cls, TrainConfig_cls):
        super().__init__()
        self.patchtst_cfg = class_to_dict(PatchTSTConfig_cls)
        self.model_cfg = class_to_dict(ModelConfig_cls)
        self.train_cfg = class_to_dict(TrainConfig_cls)
        self.save_hyperparameters(
            {
                "patchtst": self.patchtst_cfg,
                "model": self.model_cfg,
                "train": self.train_cfg,
            }
        )

        # Loss configuration
        self.loss_fn = nn.MSELoss(reduction='none')

        configs = _merge_patchtst_configs(self.patchtst_cfg, self.model_cfg, self.train_cfg)
        self.model = PatchTSTModel(configs=configs)

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
        x = self._flatten_channels(x.float())   # [B, L, enc_in]
        y = self._flatten_channels(y.float())   # [B, L, c_out] (y is already [B,L,63])

        y_pred = self.model(x)                 # [B, L, c_out]

        # print(f"\n\n\nx_shape: {x.shape}")
        # print(f"y_pred shape: {y_pred.shape}, y shape: {y.shape}\n\n\n")  # Debugging shapes

        loss = self.loss_fn(y_pred, y).mean()  # Compute MSE loss and average over all elements
        mae = F.l1_loss(y_pred, y)

        self.log(f"{step_type}/loss", loss, on_step=(step_type == "train"), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{step_type}/mae", mae, on_step=(step_type == "train"), on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._common_step(batch, "val")

    def configure_optimizers(self):
        lr = float(self.train_cfg.get("lr", 1e-3))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-2))
        warmup_ratio = float(self.train_cfg.get("warmup_ratio", 0.0))
        pretrain_epochs = int(self.train_cfg.get("pretrain_epochs", 1))

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
        self.logger.log(f"Epoch {trainer.current_epoch + 1} duration: {duration:.2f} seconds", log_type="log")
