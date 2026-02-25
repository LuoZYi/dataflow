# dataflow_train/utils/vis_semantic.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def tensor_to_uint8_img(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    if x.max() <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def overlay_mask(img_uint8: np.ndarray, mask_bool: np.ndarray, color=(255, 0, 0), alpha=0.55) -> np.ndarray:
    img = img_uint8.astype(np.float32).copy()
    m = mask_bool.astype(bool)
    if m.ndim == 2:
        m = m[..., None]
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    img[m[..., 0]] = img[m[..., 0]] * (1 - alpha) + color_arr * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def save_side_by_side(
    img_u8: np.ndarray,
    gt: np.ndarray,
    pred_prob: np.ndarray,
    out_path: str,
    thr: float = 0.5,
    max_side: int = 1600,
):
    gt = gt.astype(bool)
    pred = (pred_prob >= thr)

    gt_vis = overlay_mask(img_u8, gt, color=(0, 255, 0), alpha=0.55)
    pr_vis = overlay_mask(img_u8, pred, color=(255, 0, 0), alpha=0.55)

    gt_bw = np.stack([(gt.astype(np.uint8) * 255)] * 3, axis=-1)
    pr_bw = np.stack([(pred.astype(np.uint8) * 255)] * 3, axis=-1)

    panels = [img_u8, gt_vis, pr_vis, gt_bw, pr_bw]
    panel_imgs = [Image.fromarray(p) for p in panels]

    def downscale(im: Image.Image) -> Image.Image:
        w, h = im.size
        s = max(w, h)
        if s <= max_side:
            return im
        scale = max_side / float(s)
        nw, nh = int(w * scale), int(h * scale)
        return im.resize((nw, nh), Image.BILINEAR)

    panel_imgs = [downscale(im) for im in panel_imgs]

    W = sum(im.size[0] for im in panel_imgs)
    H = max(im.size[1] for im in panel_imgs)
    canvas = Image.new("RGB", (W, H))
    x = 0
    for im in panel_imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
