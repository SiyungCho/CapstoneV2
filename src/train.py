
import argparse
from utils import *

from config import (
    data_dir
)
from dataloader import EITDataModule

parser = argparse.ArgumentParser()
# parser.add_argument() #add needed args
args = parser.parse_args() 

device = set_device()

def main():
    data_module = EITDataModule(data_dir=data_dir, batch_size=32, num_workers=4)
    return

if __name__ == "__main__":
    args.device = device
    print(f"Using device: {args.device}")

    main()