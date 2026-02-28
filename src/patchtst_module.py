import time
from types import SimpleNamespace
import os
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from patchtst.patchtst import PatchTSTModel
from utils import class_to_dict
from config import HAND_CONNECTIONS


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

class LossHistory(L.Callback):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.train_losses = []  # index = epoch (0-based)
        self.val_losses = []    # index = epoch (0-based)

    def _to_float(self, x):
        try:
            return float(x.detach().cpu()) if hasattr(x, "detach") else float(x)
        except Exception:
            return None

    def on_train_epoch_end(self, trainer, pl_module):
        # Prefer epoch-aggregated value when available
        v = trainer.callback_metrics.get("train/loss_epoch", None)
        if v is None:
            v = trainer.callback_metrics.get("train/loss", None)
        v = self._to_float(v)
        if v is not None:
            # ensure list long enough
            while len(self.train_losses) <= trainer.current_epoch:
                self.train_losses.append(None)
            self.train_losses[trainer.current_epoch] = v

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val/loss", None)
        if v is None:
            v = trainer.callback_metrics.get("val/loss_epoch", None)
        v = self._to_float(v)
        if v is not None:
            while len(self.val_losses) <= trainer.current_epoch:
                self.val_losses.append(None)
            self.val_losses[trainer.current_epoch] = v

class QualitativeVisualizer(L.Callback):
    def __init__(self, logger, every_n_epochs = 25, num_frames = 6, sample_index = 0, out_dirname = "qualitative"):
        super().__init__()
        self.logger = logger
        self.every_n_epochs = int(max(1, every_n_epochs))
        self.num_frames = int(max(1, num_frames))
        self.sample_index = int(max(0, sample_index))
        self.out_dirname = str(out_dirname)

        self.out_dir = os.path.join(self.logger.log_dir, self.out_dirname)
        os.makedirs(self.out_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_n_epochs != 0:
            return

        batch = self._get_one_val_batch(trainer)
        x, y = batch[0], batch[1]
        self._run_and_save(trainer, pl_module, x, y, epoch)

    def _get_one_val_batch(self, trainer):
        dm = getattr(trainer, "datamodule", None)
        dl = dm.val_dataloader()
        dl = dl[0]
        return next(iter(dl))

    @torch.no_grad()
    def _run_and_save(self, trainer, pl_module, x, y, epoch):
        was_training = pl_module.training
        pl_module.eval()

        device = pl_module.device
        x = x.to(device)
        y = y.to(device)
        y_pred = pl_module(x)
        #X shape: torch.Size([64, 100, 40, 3]), y shape: torch.Size([64, 100, 63]), y_pred shape: torch.Size([64, 100, 63])

        bsz = int(y.shape[0])
        si = min(self.sample_index, max(0, bsz - 1))
        gt = y[si].detach().cpu().float().numpy()      # [L, 63]
        pr = y_pred[si].detach().cpu().float().numpy() # [L, 63]
        #Selected sample index: 0, gt shape: (100, 63), pred shape: (100, 63)

        L_ = int(gt.shape[0])
        gt = gt.reshape(L_, 21, 3)
        pr = pr.reshape(L_, 21, 3)

        frame_ids = np.linspace(0, max(0, L_ - 1), num=min(self.num_frames, L_), dtype=int)

        fp1 = os.path.join(self.out_dir, f"handpose_demo_epoch{epoch:03d}.png")
        self._plot_handpose_montage(gt, pr, frame_ids, fp1)

        fp2 = os.path.join(self.out_dir, f"error_timeseries_epoch{epoch:03d}.png")
        self._plot_error_timeseries(gt, pr, fp2)

        self.logger.log(f"[QualitativeVisualizer] saved: {os.path.relpath(fp1, self.logger.log_dir)}", log_type="log")
        self.logger.log(f"[QualitativeVisualizer] saved: {os.path.relpath(fp2, self.logger.log_dir)}", log_type="log")
        if was_training:
            pl_module.train()


    #fix functions below

    def _plot_skeleton_3d(self, ax, joints_xyz: np.ndarray, title: str = ""):
        xs, ys, zs = joints_xyz[:, 0], joints_xyz[:, 1], joints_xyz[:, 2]
        ax.scatter(xs, ys, zs, s=10)
        for a, b in HAND_CONNECTIONS:
            if a < joints_xyz.shape[0] and b < joints_xyz.shape[0]:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=-60)

    def _set_equal_3d_limits(self, ax, pts: np.ndarray):
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        span = float((maxs - mins).max())
        span = span if span > 0 else 1.0
        half = span / 2.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

    def _plot_handpose_montage(self, gt: np.ndarray, pred: np.ndarray, frame_ids: Sequence[int], save_path: str):
        n = len(frame_ids)
        cols, rows = n, 2

        fig = plt.figure(figsize=(max(10, 2.2 * cols), 5.0))
        pts = np.concatenate([gt[frame_ids].reshape(-1, 3), pred[frame_ids].reshape(-1, 3)], axis=0)

        for i, t in enumerate(frame_ids):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            self._plot_skeleton_3d(ax, gt[t], title=f"GT t={t}")
            self._set_equal_3d_limits(ax, pts)

            ax2 = fig.add_subplot(rows, cols, cols + i + 1, projection="3d")
            self._plot_skeleton_3d(ax2, pred[t], title=f"Pred t={t}")
            self._set_equal_3d_limits(ax2, pts)

        fig.suptitle("Hand Pose: Ground Truth (top) vs Prediction (bottom)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(save_path, dpi=160)
        plt.close(fig)

    def _plot_error_timeseries(self, gt: np.ndarray, pred: np.ndarray, save_path: str):
        per_joint = np.linalg.norm(pred - gt, axis=-1)  # [L, J]
        mean_err = per_joint.mean(axis=1)               # [L]

        key = [0, 4, 8, 12, 16, 20]  # wrist + fingertips
        key = [k for k in key if k < per_joint.shape[1]]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mean_err, label="mean joint L2")
        for k in key:
            ax.plot(per_joint[:, k], label=f"joint {k} L2", alpha=0.8)
        ax.set_title("Prediction Error Over Time")
        ax.set_xlabel("timestep")
        ax.set_ylabel("L2 error")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)