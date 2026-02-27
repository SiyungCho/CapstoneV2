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
    stride = 1

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
    max_epochs = 150

    # optim
    lr = 1e-3
    weight_decay = 1e-2
    warmup_ratio = 0.05
    pretrain_epochs = 5
    early_stop_patience = 10

    #model input/output dims from dataloader output, Sample x shape: (L, 40, 3), Sample y shape: (L, 63)
    #input dim (B,L,40,3) -> (B,L,120) after flattening last two dims, output dim (B,L,63)
    enc_in = 120
    target_dim = 63

    # loss
    loss = "mse"

    # logging / checkpoints
    log_dir = "./logs"
    ckpt_dir = "./checkpoints"

    # lightning precision: "32-true", "16-mixed", "bf16-mixed" (depends on GPU support)
    precision = "32-true"


# --------------- patchtst defaults ---------------

@dataclass
class PatchTSTConfig:
    # data / io
    enc_in = 1          # number of input channels (variables)
    seq_len = 100       # context window length
    pred_len = 100      # prediction window length (output sequence length)

    # transformer
    e_layers = 3
    n_heads = 8
    d_model = 128
    d_ff = 256
    dropout = 0.1
    fc_dropout = 0.1
    head_dropout = 0.0

    # head
    individual = False

    # patching
    patch_len = 16
    stride = 8
    padding_patch = None 

    # RevIN
    revin = True
    affine = True
    subtract_last = False

    # decomposition
    decomposition = False
    kernel_size = 25


# cfg = PatchTSTConfig(
#             enc_in=getattr(hparams_ns, "enc_in", 1),
#             seq_len=getattr(hparams_ns, "seq_len", 100),
#             pred_len=getattr(hparams_ns, "pred_len", getattr(hparams_ns, "seq_len", 100)),
#             e_layers=getattr(hparams_ns, "e_layers", 3),
#             n_heads=getattr(hparams_ns, "n_heads", 8),
#             d_model=getattr(hparams_ns, "d_model", 128),
#             d_ff=getattr(hparams_ns, "d_ff", 256),
#             dropout=getattr(hparams_ns, "dropout", 0.1),
#             fc_dropout=getattr(hparams_ns, "fc_dropout", 0.1),
#             head_dropout=getattr(hparams_ns, "head_dropout", 0.0),
#             individual=getattr(hparams_ns, "individual", False),
#             patch_len=getattr(hparams_ns, "patch_len", 16),
#             stride=getattr(hparams_ns, "stride", 8),
#             padding_patch=getattr(hparams_ns, "padding_patch", None),
#             revin=getattr(hparams_ns, "revin", True),
#             affine=getattr(hparams_ns, "affine", True),
#             subtract_last=getattr(hparams_ns, "subtract_last", False),
#             decomposition=getattr(hparams_ns, "decomposition", False),
#             kernel_size=getattr(hparams_ns, "kernel_size", 25),
#         )