import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import lightning as L
from torch.utils.data import DataLoader

class EITSequenceDataset(Dataset):
    def __init__(self, file_directory, set_type = "train", seq_len = 100, stride = 1, seed = 42, split=(0.7, 0.15, 0.15)):
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.set_type = set_type
        self.seed = int(seed)
        self.split = split

        self.files = [
            os.path.join(file_directory, fn)
            for fn in os.listdir(file_directory)
            if fn.endswith(".pkl")
        ]

        self._series = [] 
        self._index = [] 
        self._build_index()
        #print input and target shapes for debugging
        # print(f"Dataset initialized with {len(self._series)} series and {len(self._index)} total windows.")
        # if len(self._index) > 0:
        #     sample_entry = self._series[self._index[0][0]]
        #     print(f"Sample x shape: {sample_entry['x'].shape}, Sample y shape: {sample_entry['y'].shape}")

    def _build_index(self):
        for fp in self.files:
            payload = pd.read_pickle(fp)
            if "hand_data" not in payload or "eit_data" not in payload:
                raise ValueError(f"File {fp} does not contain expected 'hand_data' and 'eit_data' keys.")
            
            hand_df = payload["hand_data"].copy()
            eit_df = payload["eit_data"].copy()

            hand_cleaned = self.hand_cleaning(hand_df) 
            eit_cleaned = self.eit_cleaning(eit_df) 

            data, target, deltas = self.match_one_to_one_by_nearest_timestamp(
                eit_cleaned,
                hand_cleaned
            )

            data = data[:, 1:, :] #drop timestamp dims
            target = target[:, 1:] #drop timestamp dims

            series_id = len(self._series)
            entry = {
                "file": fp,
                "x": data.astype(np.float32, copy=False),
                "y": target.astype(np.float32, copy=False),
                "deltas": deltas.astype(np.int64, copy=False),
            }
            self._series.append(entry)

            N = entry["x"].shape[0]
            if N < self.seq_len:
                continue
            
            temp = []
            for start in range(0, N - self.seq_len + 1, self.stride):
                temp.append((series_id, start))
            
            #seedable shuffle of all windows, then split based on ratios 
            rng = np.random.default_rng(self.seed)
            rng.shuffle(temp)
            n_total = len(temp)
            n_train = int(n_total * self.split[0])
            n_val = int(n_total * self.split[1])
            n_test = n_total - n_train - n_val
            if self.set_type == "train":
                self._index.extend(temp[:n_train])
            elif self.set_type == "val":
                self._index.extend(temp[n_train:n_train+n_val])
            elif self.set_type == "test":
                self._index.extend(temp[n_train+n_val:])
            else:
                raise ValueError(f"Invalid set_type {self.set_type}. Expected 'train', 'val', or 'test'.")

    def eit_cleaning(self, df):
        E = "Event"
        TS = "timestamp"
        FID = "Frame_ID"
        R = "Inj"
        C = "Sense"
        V = ["Real", "Img", "Magnitude"]

        i = df.index[df[E].eq("baseline_done")].min()
        df = df.loc[i:].iloc[1:] if pd.notna(i) else df
        df = df.drop(columns=[E])

        df[TS] = df[TS].astype("int64")
        df[R] = df[R].astype("int64")
        df[C] = df[C].astype("int64")

        df["rec"] = df[R] * 5 + df[C]

        avg_ts = df.groupby(FID)[TS].agg(lambda s: (s.min() + s.max()) / 2)
        full_index = pd.MultiIndex.from_product([avg_ts.index, range(40)], names=[FID, "rec"])

        wide = (
            df.set_index([FID, "rec"])[V]
            .reindex(full_index)
            .fillna(0.0)
            .sort_index()
        )

        X = wide.to_numpy().reshape(len(avg_ts), 40, len(V))
        T = avg_ts.to_numpy()

        T3 = np.repeat(T[:, None, None], repeats=len(V), axis=2)
        X_with_time = np.concatenate([T3, X], axis=1)
        return X_with_time

    def hand_cleaning(self, df):
        E = "Event"
        i = df.index[df[E].eq("baseline_done")].min()
        df = df.loc[i:].iloc[1:] if pd.notna(i) else df
        df = df.drop(columns=[E])
        df["timestamp"] = df["timestamp"].astype("int64")
        return df.to_numpy()

    def match_one_to_one_by_nearest_timestamp(self, X1, X2, ts1_from="X1", ts2_col=0, max_delta=None):
        """
        X1: (N1, 41, 3)  timestamp at X1[:,0,0]
        X2: (N2, D)      timestamp at X2[:,0] by default
        """
        if ts1_from == "X1":
            T1 = X1[:, 0, 0].astype(np.int64)
        else:
            T1 = np.asarray(ts1_from, dtype=np.int64)

        T2 = X2[:, ts2_col].astype(np.int64)

        order2 = np.argsort(T2)
        T2s = T2[order2]
        X2s = X2[order2]

        pos = np.searchsorted(T2s, T1)
        left = np.clip(pos - 1, 0, len(T2s) - 1)
        right = np.clip(pos, 0, len(T2s) - 1)

        d_left = np.abs(T2s[left] - T1)
        d_right = np.abs(T2s[right] - T1)

        best2 = np.where(d_right < d_left, right, left)
        delta = np.minimum(d_left, d_right)

        order_pairs = np.argsort(delta)
        used2 = np.zeros(len(T2s), dtype=bool)

        keep1, keep2 = [], []
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
        dm = delta[keep1]
        return X1m, X2m, dm

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        series_id, start = self._index[idx]
        entry = self._series[series_id]

        x = entry["x"][start:start + self.seq_len]
        y = entry["y"][start:start + self.seq_len]
        deltas = entry["deltas"][start : start + self.seq_len]

        x = torch.from_numpy(x) 
        y = torch.from_numpy(y) 
        deltas = torch.from_numpy(deltas) 

        return x, y

class EITDataModule(L.LightningDataModule):
    def __init__(self, data_dir, seq_len = 100, stride = 1, batch_size = 32, num_workers = 4, pin_memory = True):
        super().__init__()
        print("DataModule initialized with data_dir:", data_dir)
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
        if stage in (None, "fit"):
            self.ds_train = EITSequenceDataset(
                file_directory=self.data_dir,
                set_type="train",
                seq_len=self.seq_len,
                stride=self.stride,
                seed=42,
            )
            self.ds_val = EITSequenceDataset(
                file_directory=self.data_dir,
                set_type="val",
                seq_len=self.seq_len,
                stride=self.stride,
                seed=42,
            )

        if stage in (None, "test"):
            self.ds_test = EITSequenceDataset(
                file_directory=self.data_dir,
                set_type="test",
                seq_len=self.seq_len,
                stride=self.stride,
                seed=42,
            )

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
