from dataclasses import dataclass


# --------------- existing hand skeleton ---------------
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
    data_dir = "/Users/frankcho/Desktop/CapstoneV2/src/data/"

    seq_len = 100
    data_stride = 1

    batch_size = 64
    num_workers = 4
    pin_memory = True


# --------------- model defaults (PatchTST) ---------------
@dataclass(frozen=True)
class ModelConfig:
    # patching (PatchTST)
    patch_len = 12
    patch_stride = 12

    # transformer
    d_model = 128
    d_ff = 256
    e_layers = 3
    n_heads = 8
    dropout = 0.1

    # RevIN
    revin = True


# --------------- training defaults ---------------
@dataclass(frozen=True)
class TrainConfig:
    seed = 42
    max_epochs = 50

    # optim
    lr = 1e-3
    weight_decay = 1e-2
    warmup_ratio = 0.05

    # loss: "mse" or "mae"
    loss = "mse"

    # logging / checkpoints
    log_dir = "./logs"
    ckpt_dir = "./checkpoints"

    # lightning precision: "32-true", "16-mixed", "bf16-mixed" (depends on GPU support)
    precision = "32-true"

    early_stop_patience = 10
