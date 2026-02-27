
import argparse
from utils import *

from config import (
    data_dir
)

parser = argparse.ArgumentParser()
# parser.add_argument() #add needed args
args = parser.parse_args() 

device = set_device()

def main():
    return

if __name__ == "__main__":
    args.device = device
    print(f"Using device: {args.device}")

    main()