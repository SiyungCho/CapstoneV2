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
    # data_dir = "/Users/frankcho/Desktop/CapstoneV2/src/data/"
    data_dir = "/home/siyung/Desktop/CapstoneV2/src/data/"

    seq_len = 100
    stride = 1

    batch_size = 16
    num_workers = 4
    pin_memory = False

# --------------- model defaults ---------------
@dataclass(frozen=True)
class ModelConfig:
    dummy = True

# --------------- training defaults ---------------
@dataclass(frozen=True)
class TrainConfig:
    seed = 42
    max_epochs = 300

    # optim
    lr = 1e-3
    weight_decay = 1e-2
    warmup_ratio = 0.05
    pretrain_epochs = 10
    early_stop_patience = 10

    #model input/output dims from dataloader output, Sample x shape: (L, 40, 3), Sample y shape: (L, 63)
    #input dim (B,L,40,3) -> (B,L,120) after flattening last two dims, output dim (B,L,63)
    enc_in = 120
    target_dim = 63

    # loss
    loss = "mse"

    # logging / checkpoints
    log_dir = "./logs"

    # lightning precision: "32-true", "16-mixed", "bf16-mixed" (depends on GPU support)
    precision = "32-true"


# --------------- patchtst defaults ---------------

@dataclass
class PatchTSTConfig:
    # data / io
    seq_len = DataConfig.seq_len       # context window length
    pred_len = DataConfig.seq_len      # prediction window length (output sequence length)

    # transformer
    e_layers = 6
    n_heads = 16
    d_model = 256
    d_ff = 512
    dropout = 0.5
    fc_dropout = 0.5
    head_dropout = 0.5

    # head
    individual = False

    # patching
    patch_len = 32
    stride = 16
    padding_patch = None 

    # RevIN
    revin = True
    affine = True
    subtract_last = False

    # decomposition
    decomposition = True
    kernel_size = 25