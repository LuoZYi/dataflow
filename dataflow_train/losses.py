import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def dice_score(logits: torch.Tensor, y: torch.Tensor, eps=1e-6) -> torch.Tensor:
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p*y).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean()

def loss_bce_dice(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p*y).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    dice = 1 - ((2*inter + 1e-6) / (union + 1e-6)).mean()
    return bce + dice

def save_debug_overlays(x_chw: torch.Tensor, logits_chw: torch.Tensor, out_png):
    img = (x_chw.detach().cpu().numpy().transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
    pm = (torch.sigmoid(logits_chw[0]).detach().cpu().numpy() > 0.5)
    if pm.any():
        red = np.array([255,0,0], dtype=np.uint8)
        img[pm] = (img[pm].astype(np.float32)*0.4 + red.astype(np.float32)*0.6).astype(np.uint8)
    Image.fromarray(img).save(out_png)
