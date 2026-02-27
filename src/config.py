from dataclasses import dataclass


# --------------- existing hand skeleton (kept) ---------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------- data defaults ---------------
@dataclass(frozen=True)
class DataConfig:
    # IMPORTANT: change this to your dataset folder if needed
    data_dir: str = "/home/siyung/Desktop/CapstoneV2/src/data"

    # sequence windowing (must match how you want to train)
    seq_len: int = 100
    data_stride: int = 1

    # dataloader
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True


# --------------- model defaults (PatchTST) ---------------
@dataclass(frozen=True)
class ModelConfig:
    # patching (PatchTST)
    patch_len: int = 12
    patch_stride: int = 12

    # transformer
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 3
    n_heads: int = 8
    dropout: float = 0.1

    # RevIN
    revin: bool = True


# --------------- training defaults ---------------
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    max_epochs: int = 50

    # optim
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05

    # loss: "mse" or "mae"
    loss: str = "mse"

    # logging / checkpoints
    log_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"

    # lightning precision: "32-true", "16-mixed", "bf16-mixed" (depends on GPU support)
    precision: str = "32-true"

    early_stop_patience: int = 10
