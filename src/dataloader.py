import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import lightning as L
from torch.utils.data import DataLoader

class EITSequenceDataset(Dataset):
    def __init__(self, file_directory, set_type="train", seq_len=100, stride=1):
        self.seq_len = seq_len
        self.stride = stride
       
        self.files = []
        for fn in os.listdir(file_directory):
            if set_type == "train" and fn.startswith("RR-U1"):
                self.files.append(os.path.join(file_directory, fn))
            elif set_type == "validation" and fn.startswith("RR-U2"):
                self.files.append(os.path.join(file_directory, fn))
            elif set_type == "test" and fn.startswith("RR-U3"):
                self.files.append(os.path.join(file_directory, fn))

        self.sessions = [] 
        self.index = [] 

        for fp in self.files:
            df = pd.read_pickle(fp)

            eit = np.stack(df["eit_data"].tolist()).astype(np.float32)
            lab = np.stack(df["mphands_data"].tolist()).astype(np.float32)

            eit = eit - eit.mean(axis=0, keepdims=True)
            lab = lab * 10.0

            T = eit.shape[0]
            # print(f"Loaded file {fp} with {T} time steps.")
            # print("sequence length:", self.seq_len)
            if T < self.seq_len:
                continue

            sid = len(self.sessions)
            self.sessions.append((eit, lab))

            for start in range(0, T - self.seq_len + 1, self.stride):
                self.index.append((sid, start))

        print(f"Loaded {len(self.sessions)} sessions, {len(self.index)} windows.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, start = self.index[idx]
        eit, lab = self.sessions[sid]

        x = eit[start : start + self.seq_len] 
        y = lab[start : start + self.seq_len] 

        y = y.reshape(self.seq_len, 21, 3)

        # return torch.from_numpy(x), torch.from_numpy(y)
        return torch.from_numpy(x)

class EITDataModule(L.LightningDataModule):
    def __init__(self, data_dir, seq_len = 100, stride = 1, batch_size = 32, num_workers = 4, pin_memory = True):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def setup(self, stage=None):
        self.ds_train = EITSequenceDataset(file_directory=self.data_dir, set_type="train", seq_len=self.seq_len, stride=self.stride)
        self.ds_val = EITSequenceDataset(file_directory=self.data_dir, set_type="validation", seq_len=self.seq_len, stride=self.stride)
        self.ds_test = EITSequenceDataset(file_directory=self.data_dir, set_type="test", seq_len=self.seq_len, stride=self.stride)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )
