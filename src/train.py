import argparse
import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import torch

from dataloader import EITDataModule
from patchtst.module import PatchTSTLightningModule
from config import TrainConfig, ModelConfig, DataConfig


def infer_enc_in() -> int:
    # EIT input from dataset is (seq_len, 40, 3) => flatten => 120 channels
    return 40 * 3


def main():
    parser = argparse.ArgumentParser(description="Train PatchTST on EIT -> Hand sequences")

    # Data args
    parser.add_argument("--data_dir", type=str, default=DataConfig.data_dir)
    parser.add_argument("--seq_len", type=int, default=DataConfig.seq_len)
    parser.add_argument("--data_stride", type=int, default=DataConfig.data_stride)
    parser.add_argument("--batch_size", type=int, default=DataConfig.batch_size)
    parser.add_argument("--num_workers", type=int, default=DataConfig.num_workers)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=DataConfig.pin_memory)

    # Model / PatchTST args
    parser.add_argument("--patch_len", type=int, default=ModelConfig.patch_len)
    parser.add_argument("--patch_stride", type=int, default=ModelConfig.patch_stride)
    parser.add_argument("--d_model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--d_ff", type=int, default=ModelConfig.d_ff)
    parser.add_argument("--e_layers", type=int, default=ModelConfig.e_layers)
    parser.add_argument("--n_heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--revin", action=argparse.BooleanOptionalAction, default=ModelConfig.revin)

    # Optim / training args
    parser.add_argument("--max_epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--precision", type=str, default=TrainConfig.precision)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log_dir", type=str, default=TrainConfig.log_dir)
    parser.add_argument("--ckpt_dir", type=str, default=TrainConfig.ckpt_dir)
    parser.add_argument("--early_stop_patience", type=int, default=TrainConfig.early_stop_patience)

    args = parser.parse_args()

    # Reproducibility
    L.seed_everything(args.seed, workers=True)

    # DataModule
    dm = EITDataModule(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.data_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # PatchTST hparams expected by PatchTSTLightningModule
    enc_in = infer_enc_in()
    hparams = dict(
        # PatchTSTConfig fields
        enc_in=enc_in,
        seq_len=args.seq_len,
        pred_len=args.seq_len,  # predict a value per timestep for the same window length
        patch_len=args.patch_len,
        stride=args.patch_stride,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        revin=args.revin,

        # Optim
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # Loss (optional: "mse" or "mae")
        loss=TrainConfig.loss,
    )

    model = PatchTSTLightningModule(hparams)

    # Logging + callbacks
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    logger = CSVLogger(save_dir=args.log_dir, name="patchtst")

    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="patchtst-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=args.early_stop_patience,
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # Accelerator selection
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, es_cb, lr_cb],
        log_every_n_steps=25,
        deterministic=True,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=dm)

    # Optionally run test on best checkpoint
    if ckpt_cb.best_model_path:
        trainer.test(model=None, datamodule=dm, ckpt_path=ckpt_cb.best_model_path)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
