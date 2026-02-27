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
            if not fn.endswith(".pkl"):
                continue
            payload = pd.read_pickle(os.path.join(file_directory, fn))
            hand_df = payload["hand_data"].copy()
            eit_df = payload["eit_data"].copy()

            hand_cleaned = self.hand_cleaning(hand_df)
            eit_cleaned = self.eit_cleaning(eit_df)
            data, target, deltas = self.match_one_to_one_by_nearest_timestamp(eit_cleaned, hand_cleaned)

            eit_ts  = data[:, 0, 0].copy()
            hand_ts = target[:, 0].copy()

            data   = data[:, 1:, :]
            target = target[:, 1:]


    def eit_cleaning(self, df):
        col = "Event" 
        TS  = "timestamp"
        FID = "Frame_ID"
        R   = "Inj"
        C   = "Sense"

        V = ["Real", "Img", "Magnitude"]

        i = df.index[df[col].eq("baseline_done")].min() 
        df = df.loc[i:].iloc[1:] if pd.notna(i) else df 
        df = df.drop(columns=["Event"])
        df["timestamp"] = df["timestamp"].astype("int64")

        df[R] = df[R].astype(int)
        df[C] = df[C].astype(int)

        # record index 0..39 (8*5=40) based on row/col
        df["rec"] = df[R] * 5 + df[C]

        # average timestamp per frame (common definition: midpoint)
        avg_ts = df.groupby(FID)[TS].agg(lambda s: (s.min() + s.max()) / 2)

        # build a complete frame x rec index so every frame has exactly 40 records
        full_index = pd.MultiIndex.from_product([avg_ts.index, range(40)], names=[FID, "rec"])

        # keep only what we need, align to full grid, fill missing
        wide = (
            df.set_index([FID, "rec"])[V]
            .reindex(full_index)
            .fillna(0.0)
            .sort_index()
        )

        # tensor (num_frames, 40, 3)
        X = wide.to_numpy().reshape(len(avg_ts), 40, len(V))

        # timestamps (num_frames,)
        T = avg_ts.to_numpy()

        T3 = np.repeat(T[:, None, None], repeats=len(V), axis=2)  # (N,1,3)
        X_with_time = np.concatenate([T3, X], axis=1)             # (N,41,3)

        return X_with_time

    def hand_cleaning(self, df):
        col = "Event" 

        i = df.index[df[col].eq("baseline_done")].min() 
        df = df.loc[i:].iloc[1:] if pd.notna(i) else df 
        df = df.drop(columns=["Event"])
        df["timestamp"] = df["timestamp"].astype("int64")
        X = df.to_numpy()
        return X

    def match_one_to_one_by_nearest_timestamp(self, X1, X2, ts1_from="X1", ts2_col=0, max_delta=None):
        """
        X1: (N1, 41, 3)  (timestamp assumed at X1[:,0,0] if ts1_from="X1")
        X2: (N2, 64)     (timestamp assumed at X2[:, ts2_col])

        max_delta: optional, in same units as timestamps. If set, drops pairs farther than this.
        Returns: (X1_matched, X2_matched, deltas)
        """
        # timestamps
        if ts1_from == "X1":
            T1 = X1[:, 0, 0].astype(np.int64)
        else:
            T1 = np.asarray(ts1_from, dtype=np.int64)

        T2 = X2[:, ts2_col].astype(np.int64)

        # sort X2 by timestamp for fast nearest lookup
        order2 = np.argsort(T2)
        T2s = T2[order2]
        X2s = X2[order2]

        # nearest candidate in sorted T2 for each T1: compare neighbor left/right
        pos = np.searchsorted(T2s, T1)
        left  = np.clip(pos - 1, 0, len(T2s) - 1)
        right = np.clip(pos,     0, len(T2s) - 1)

        d_left  = np.abs(T2s[left]  - T1)
        d_right = np.abs(T2s[right] - T1)

        best2 = np.where(d_right < d_left, right, left)   # chosen X2 index (in sorted space)
        delta = np.minimum(d_left, d_right)

        # enforce one-to-one by taking smallest deltas first
        order_pairs = np.argsort(delta)   # X1 indices sorted by closeness
        used2 = np.zeros(len(T2s), dtype=bool)

        keep1 = []
        keep2 = []
        for i in order_pairs:
            j = best2[i]
            if used2[j]:
                continue
            if max_delta is not None and delta[i] > max_delta:
                continue
            used2[j] = True
            keep1.append(i)
            keep2.append(j)

        keep1 = np.array(keep1, dtype=int)
        keep2 = np.array(keep2, dtype=int)

        X1m = X1[keep1]
        X2m = X2s[keep2]
        dm  = delta[keep1]

        return X1m, X2m, dm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return

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
