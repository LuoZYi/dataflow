# dataflow_train/utils/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch


@torch.no_grad()
def dice_per_sample(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    prob: [B,1,H,W] in [0,1]
    target: [B,1,H,W] float {0,1}
    return: [B]
    """
    pred = (prob > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (den + eps)


def stats(arr: List[float]) -> Tuple[float, float, float]:
    if not arr:
        return 0.0, 0.0, 0.0
    a = np.array(arr, dtype=np.float32)
    return float(a.mean()), float(np.quantile(a, 0.5)), float(np.quantile(a, 0.9))


@dataclass
class ValStats:
    val_loss: float
    dice_all_mean: float
    dice_all_p50: float
    dice_all_p90: float
    dice_nonempty_mean: float
    dice_nonempty_p50: float
    dice_nonempty_p90: float
    empty_gt_rate: float
    empty_pred_rate_on_empty_gt: float
    fg_frac_mean: float
    pred_fg_frac_mean: float
