import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from dataloader import EITDataModule
from patchtst_module import PatchTSTLightningModule
from config import TrainConfig, ModelConfig, DataConfig, PatchTSTConfig
from logger import JsonLogger
from utils import set_device, class_to_dict

device = set_device()
logger = JsonLogger(log_dir=TrainConfig.log_dir)
logger.log(class_to_dict(ModelConfig), log_type="model_arguments", skip_if_exists=True)
logger.log(class_to_dict(DataConfig), log_type="data_arguments", skip_if_exists=True)
logger.log(class_to_dict(TrainConfig), log_type="train_arguments", skip_if_exists=True)
logger.log(class_to_dict(PatchTSTConfig), log_type="patchtstmodel_arguments", skip_if_exists=True)

def main():
    L.seed_everything(TrainConfig.seed, workers=True)
    dm = EITDataModule(**class_to_dict(DataConfig))
    # dm.setup(stage="fit")
    model = PatchTSTLightningModule(PatchTSTConfig, ModelConfig, TrainConfig)

    # ckpt_cb = ModelCheckpoint(
    #     dirpath=args.ckpt_dir,
    #     filename="patchtst-{epoch:03d}-{val_loss:.4f}",
    #     monitor="val/loss",
    #     mode="min",
    #     save_top_k=1,
    #     save_last=True,
    # )
    # es_cb = EarlyStopping(
    #     monitor="val/loss",
    #     mode="min",
    #     patience=args.early_stop_patience,
    #     verbose=True,
    # )
    # lr_cb = LearningRateMonitor(logging_interval="step")

    # # Accelerator selection
    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # devices = 1

    # trainer = L.Trainer(
    #     max_epochs=args.max_epochs,
    #     accelerator=accelerator,
    #     devices=devices,
    #     precision=args.precision,
    #     logger=logger,
    #     callbacks=[ckpt_cb, es_cb, lr_cb],
    #     log_every_n_steps=25,
    #     deterministic=True,
    #     enable_checkpointing=True,
    # )

    # trainer.fit(model, datamodule=dm)

    # # Optionally run test on best checkpoint
    # if ckpt_cb.best_model_path:
    #     trainer.test(model=None, datamodule=dm, ckpt_path=ckpt_cb.best_model_path)
    # else:
    #     trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
