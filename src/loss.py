import torch
import torch.nn as nn

from config import HAND_CONNECTIONS

def _to_B_T_J_3(x: torch.Tensor, num_joints: int = 21) -> torch.Tensor:
    """
    Accepts:
      - (B,T,21,3)
      - (B,T,63)
    Returns:
      - (B,T,21,3)
    """
    if x.dim() == 4:
        # (B,T,J,3)
        if x.shape[-2] != num_joints or x.shape[-1] != 3:
            raise ValueError(f"Expected (B,T,{num_joints},3), got {tuple(x.shape)}")
        return x
    if x.dim() == 3:
        # (B,T,63) -> (B,T,21,3)
        B, T, D = x.shape
        if D != num_joints * 3:
            raise ValueError(f"Expected last dim {num_joints*3}, got {D}")
        return x.view(B, T, num_joints, 3)
    raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D with shape {tuple(x.shape)}")


class MPJPELoss(nn.Module):
    """
    Mean Per Joint Position Error (scalar).
    Works with pred/tgt shaped (B,T,21,3) or (B,T,63).
    """
    def __init__(self, num_joints: int = 21, eps: float = 1e-8):
        super().__init__()
        self.num_joints = num_joints
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = _to_B_T_J_3(predictions, self.num_joints)
        tgt  = _to_B_T_J_3(targets, self.num_joints)

        diff = pred - tgt  # (B,T,J,3)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps)  # (B,T,J)
        return dist.mean()  # scalar


def bone_length_loss(pred: torch.Tensor,
                     tgt: torch.Tensor,
                     edges=HAND_CONNECTIONS,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    pred, tgt: (B,T,21,3) or (B,T,63)
    Returns scalar.
    """
    pred = _to_B_T_J_3(pred, 21)
    tgt  = _to_B_T_J_3(tgt, 21)

    # Vectorize edges
    a = torch.tensor([e[0] for e in edges], device=pred.device, dtype=torch.long)
    b = torch.tensor([e[1] for e in edges], device=pred.device, dtype=torch.long)

    pred_vec = pred[:, :, a, :] - pred[:, :, b, :]   # (B,T,E,3)
    tgt_vec  = tgt[:, :, a, :] - tgt[:, :, b, :]     # (B,T,E,3)

    pred_len = torch.sqrt((pred_vec * pred_vec).sum(dim=-1) + eps)  # (B,T,E)
    tgt_len  = torch.sqrt((tgt_vec  * tgt_vec ).sum(dim=-1) + eps)  # (B,T,E)

    return (pred_len - tgt_len).abs().mean()  # scalar


def temporal_smoothness_loss(pred: torch.Tensor,
                             vel_weight: float = 1.0,
                             acc_weight: float = 1.0,
                             eps: float = 1e-8) -> torch.Tensor:
    """
    pred: (B,T,21,3) or (B,T,63)
    Returns scalar.
    """
    pred = _to_B_T_J_3(pred, 21)

    # Guard for short sequences
    T = pred.shape[1]
    if T < 2:
        return pred.new_tensor(0.0)

    v = pred[:, 1:] - pred[:, :-1]          # (B,T-1,21,3)
    v_mag = torch.sqrt((v * v).sum(dim=-1) + eps)     # (B,T-1,21)

    if T < 3:
        return vel_weight * v_mag.mean()

    a = v[:, 1:] - v[:, :-1]                # (B,T-2,21,3)
    a_mag = torch.sqrt((a * a).sum(dim=-1) + eps)     # (B,T-2,21)

    return vel_weight * v_mag.mean() + acc_weight * a_mag.mean()


class WeightedMPJPELoss(nn.Module):
    """
    Weighted MPJPE (scalar).
    Normalization fixed: divides by sum(weights), not mean(weights).
    """
    def __init__(self, joint_weights=None, num_joints: int = 21, eps: float = 1e-8):
        super().__init__()
        self.num_joints = num_joints
        self.eps = eps

        if joint_weights is None:
            w = torch.ones(num_joints)
            w[0] = 2.0                      # wrist
            for j in [1, 5, 9, 13, 17]:      # palm anchors (MediaPipe-ish)
                w[j] = 1.5
            joint_weights = w

        joint_weights = torch.as_tensor(joint_weights, dtype=torch.float32)
        if joint_weights.numel() != num_joints:
            raise ValueError(f"joint_weights must have length {num_joints}, got {joint_weights.numel()}")
        self.register_buffer("w", joint_weights)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = _to_B_T_J_3(pred, self.num_joints)
        tgt  = _to_B_T_J_3(tgt, self.num_joints)

        diff = pred - tgt
        dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps)   # (B,T,J)

        w = self.w.view(1, 1, self.num_joints)                    # (1,1,J)
        weighted = (dist * w).sum(dim=-1) / (w.sum(dim=-1) + self.eps)  # (B,T)
        return weighted.mean()  # scalar


class HandPoseLoss(nn.Module):
    def __init__(self, lambda_bone=0.05, lambda_smooth=0.01, joint_weights=None):
        super().__init__()
        self.mpjpe = WeightedMPJPELoss(joint_weights=joint_weights)
        self.lambda_bone = lambda_bone
        self.lambda_smooth = lambda_smooth

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # pred, tgt: (B,T,21,3) (your case) OR (B,T,63)
        loss_mpjpe  = self.mpjpe(pred, tgt)
        loss_bone   = bone_length_loss(pred, tgt)
        loss_smooth = temporal_smoothness_loss(pred, vel_weight=1.0, acc_weight=1.0)

        return loss_mpjpe + self.lambda_bone * loss_bone + self.lambda_smooth * loss_smooth