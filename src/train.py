import gc
from fvcore.nn import FlopCountAnalysis
import time
import matplotlib.pyplot as plt

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from dataloader import EITDataModule
from patchtst_module import PatchTSTLightningModule, EpochTimer
from config import TrainConfig, ModelConfig, DataConfig, PatchTSTConfig
from logger import JsonLogger
from utils import set_device, class_to_dict

device = set_device()
logger = JsonLogger(log_dir=TrainConfig.log_dir)
logger.log(class_to_dict(ModelConfig), log_type="model_arguments", skip_if_exists=True)
logger.log(class_to_dict(DataConfig), log_type="data_arguments", skip_if_exists=True)
logger.log(class_to_dict(TrainConfig), log_type="train_arguments", skip_if_exists=True)
logger.log(class_to_dict(PatchTSTConfig), log_type="patchtstmodel_arguments", skip_if_exists=True)

def generate_train_summary(training_duration, logger, total_flops, flops_analyzer, epoch_timer_callback):
    hours, rem = divmod(training_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.log("\n--- Training Summary ---", log_type="log")
    logger.log(f"Total training runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}", log_type="log")
    
    if total_flops > 0 and flops_analyzer is not None:
        logger.log(f"Total FLOPs per forward pass: {total_flops / 1e9:.2f} GFLOPs", log_type="log")
        logger.log("--- FLOPs Breakdown by Module ---", log_type="log")
        logger.log(flops_analyzer.by_module(), log_type="log")
        logger.log("---------------------------------", log_type="log")
    logger.log("------------------------\n", log_type="log")
   
    if epoch_timer_callback.epoch_times:
        logger.log("Generating plot for epoch training times...", log_type="log")
        plt.figure(figsize=(10, 6))
        num_epochs_completed = range(1, len(epoch_timer_callback.epoch_times) + 1)
        plt.plot(num_epochs_completed, epoch_timer_callback.epoch_times, marker='o', linestyle='-')
        plt.title('Training Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.xticks(list(num_epochs_completed))
        plt.tight_layout()
        plt.savefig(logger.log_dir + "/epoch_times.png")
        plt.close() # Close the figure to free up memory
        logger.log(f"Epoch times plot saved to epoch_times.png", log_type="log")

def train(TrainConfig, logger, data_module, model, visualizer_callback, total_flops, flops_analyzer):
    epoch_timer_callback = EpochTimer(logger)

    ckpt_callback = ModelCheckpoint(
        dirpath=TrainConfig.log_dir + "/checkpoints",
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

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = L.Trainer(
        max_epochs=TrainConfig.pretrain_epochs,
        accelerator=accelerator, 
        devices=devices,
        precision=TrainConfig.precision,
        # logger=logger,
        strategy='auto',
        log_every_n_steps=20,
        callbacks=[
            ckpt_callback,
            es_callback,
            LearningRateMonitor(logging_interval="step"),
            # visualizer_callback,
            epoch_timer_callback
        ],
        deterministic=True,
        enable_checkpointing=True,
    )

    training_duration = 0
    start_time = time.time()

    logger.log("Starting HIMAE pre-training...", log_type="log")
    trainer.fit(model, datamodule=data_module)
    logger.log("HIMAE pre-training complete.", log_type="log")

    end_time = time.time()
    training_duration = end_time - start_time
    generate_train_summary(training_duration, logger, total_flops, flops_analyzer, epoch_timer_callback)

    # # Optionally run test on best checkpoint
    # if ckpt_cb.best_model_path:
    #     trainer.test(model=None, datamodule=dm, ckpt_path=ckpt_cb.best_model_path)
    # else:
    #     trainer.test(model=model, datamodule=dm)
    return

def main():
    L.seed_everything(TrainConfig.seed, workers=True)
    dm = EITDataModule(**class_to_dict(DataConfig))
    # dm.setup(stage="fit")
    model = PatchTSTLightningModule(PatchTSTConfig, ModelConfig, TrainConfig)

    total_flops = 0
    flops_analyzer = None
    dummy_input = torch.randn(1, DataConfig.seq_len, 40, 3).to(device)
    
    flops_analyzer = FlopCountAnalysis(model.model, dummy_input)
    total_flops = flops_analyzer.total()
    logger.log(f"Total FLOPs: {total_flops}", log_type="log")

    # visualizer_callback = HiMAEReconstructionVisualizer(data_module.val_dataloader(), logger.log_dir, every_n_epochs=2)

    train(
        TrainConfig,
        logger,
        dm,
        model,
        None,
        # visualizer_callback,
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
