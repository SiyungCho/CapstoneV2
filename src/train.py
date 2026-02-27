import time
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import gc

import argparse
from utils import *

from logger import JsonLogger
from config import (
    data_dir
)
from dataloader import EITDataModule

logger = JsonLogger(log_dir="./logs")
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
args = parser.parse_args() 
logger.log(vars(args), log_type="data_arguments", skip_if_exists=True)

device = set_device()
args.device = device
logger.log(f"Using device: {args.device}", log_type="log")

def main():
    data_module = EITDataModule(data_dir=data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    del data_module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.log("CUDA cache cleared.", log_type="log")

    logger.log("Cleanup complete. Exiting.", log_type="log")
    return

if __name__ == "__main__":
    main()