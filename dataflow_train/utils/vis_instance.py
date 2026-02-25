# dataflow_train/utils/vis_instance.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def _colorize_instances(inst_map: np.ndarray) -> np.ndarray:
    """inst_map [H,W] int32 -> RGB uint8"""
    H, W = inst_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    rng = np.random.RandomState(12345)
    lut = {}
    for iid in ids:
        lut[int(iid)] = rng.randint(0, 255, size=(3,), dtype=np.uint8)
    for iid in ids:
        out[inst_map == iid] = lut[int(iid)]
    return out


def _overlay(img_u8: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    img = img_u8.astype(np.float32).copy()
    m = mask.astype(bool)
    col = np.array(color, dtype=np.float32)[None, None, :]
    img[m] = img[m] * (1 - alpha) + col * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def save_instance_side_by_side(
    img_u8: np.ndarray,
    inst_map: np.ndarray,
    nuclei_gt: np.ndarray,
    nuclei_prob: np.ndarray,
    out_path: str,
    thr: float = 0.5,
    max_side: int = 1600,
    text: str | None = None,
):
    pred = (nuclei_prob >= thr)
    inst_rgb = _colorize_instances(inst_map.astype(np.int32))

    gt_bw = (nuclei_gt.astype(np.uint8) * 255)
    pr_bw = (pred.astype(np.uint8) * 255)
    gt_bw = np.stack([gt_bw] * 3, axis=-1)
    pr_bw = np.stack([pr_bw] * 3, axis=-1)

    gt_ov = _overlay(img_u8, nuclei_gt, color=(0, 255, 0), alpha=0.45)
    pr_ov = _overlay(img_u8, pred, color=(255, 0, 0), alpha=0.45)

    panels = [img_u8, inst_rgb, gt_bw, pr_bw, gt_ov, pr_ov]
    ims = [Image.fromarray(p) for p in panels]

    def downscale(im: Image.Image) -> Image.Image:
        w, h = im.size
        s = max(w, h)
        if s <= max_side:
            return im
        scale = max_side / float(s)
        return im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    ims = [downscale(im) for im in ims]
    W = sum(im.size[0] for im in ims)
    H = max(im.size[1] for im in ims)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    if text:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.size[0], 22), fill=(0, 0, 0))
        draw.text((6, 3), text[:200], fill=(255, 255, 255))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)