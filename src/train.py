
import argparse
from utils import *

from config import (
    data_dir
)
from dataloader import EITSequenceDataset

parser = argparse.ArgumentParser()
# parser.add_argument() #add needed args
args = parser.parse_args() 

device = set_device()

def main():
    dataset = EITSequenceDataset(file_directory=data_dir, set_type="train", seq_len=100, stride=1)
    return

if __name__ == "__main__":
    args.device = device
    print(f"Using device: {args.device}")

    main()