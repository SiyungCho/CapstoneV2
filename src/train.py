import gc
from fvcore.nn import FlopCountAnalysis
import time
import matplotlib.pyplot as plt

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from dataloader import EITDataModule
from patchtst_module import PatchTSTLightningModule, EpochTimer, LossHistory, QualitativeVisualizer
from config import TrainConfig, ModelConfig, DataConfig, PatchTSTConfig
from logger import JsonLogger, CustomLightningLogger
from utils import set_device, class_to_dict

device = set_device()
logger = JsonLogger(log_dir=TrainConfig.log_dir)
logger.log(class_to_dict(ModelConfig), log_type="model_arguments", skip_if_exists=True)
logger.log(class_to_dict(DataConfig), log_type="data_arguments", skip_if_exists=True)
logger.log(class_to_dict(TrainConfig), log_type="train_arguments", skip_if_exists=True)
logger.log(class_to_dict(PatchTSTConfig), log_type="patchtstmodel_arguments", skip_if_exists=True)

def train(TrainConfig, logger, data_module, model, total_flops, flops_analyzer):
    epoch_timer_callback = EpochTimer(logger)
    loss_history_callback = LossHistory(logger)
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    ckpt_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="patchtst-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    es_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=TrainConfig.early_stop_patience,
        verbose=True,
    )

    visualizer_callback = QualitativeVisualizer()

    trainer = L.Trainer(
        max_epochs=TrainConfig.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", 
        devices=1,
        precision=TrainConfig.precision,
        logger=CustomLightningLogger(logger, flops_analyzer=flops_analyzer, total_flops=total_flops, epoch_timer_callback=epoch_timer_callback, loss_history_callback=loss_history_callback),
        strategy='auto',
        log_every_n_steps=20,
        callbacks=[
            ckpt_callback,
            es_callback,
            lr_monitor_callback,
            visualizer_callback,
            epoch_timer_callback,
            loss_history_callback,
        ],
        deterministic=True,
        enable_checkpointing=True,
    )

    logger.log("Starting PatchTST pre-training...", log_type="log")
    trainer.fit(model, datamodule=data_module)

    # # Optionally run test on best checkpoint
    # if ckpt_cb.best_model_path:
    #     trainer.test(model=None, datamodule=dm, ckpt_path=ckpt_cb.best_model_path)
    # else:
    #     trainer.test(model=model, datamodule=dm)
    return

def main():
    L.seed_everything(TrainConfig.seed, workers=True)
    dm = EITDataModule(**class_to_dict(DataConfig))
    model = PatchTSTLightningModule(PatchTSTConfig, ModelConfig, TrainConfig)

    total_flops = 0
    flops_analyzer = None
    dummy_input = torch.randn(1, DataConfig.seq_len, 120)
    
    flops_analyzer = FlopCountAnalysis(model.model, dummy_input)
    total_flops = flops_analyzer.total()
    logger.log(f"Total FLOPs: {total_flops}", log_type="log")

    train(
        TrainConfig,
        logger,
        dm,
        model,
        total_flops,
        flops_analyzer
    )

    logger.log("Cleaning up resources...", log_type="log")

    del model
    del dm

    gc.collect()
    logger.log("Garbage collection complete.", log_type="log")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.log("CUDA cache cleared.", log_type="log")

    logger.log("Cleanup complete. Exiting.", log_type="log")

if __name__ == "__main__":
    main()
