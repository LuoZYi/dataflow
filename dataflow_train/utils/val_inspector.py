# dataflow_train/utils/val_inspector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(":", "_").replace(" ", "_")


def _dice_from_prob(prob: np.ndarray, gt: np.ndarray, thr: float, eps: float = 1e-6) -> float:
    pred = prob >= thr
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (den + eps))


def _overlay(img_u8: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    img = img_u8.astype(np.float32).copy()
    m = mask.astype(bool)
    col = np.array(color, dtype=np.float32)[None, None, :]
    img[m] = img[m] * (1 - alpha) + col * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def save_side_by_side_patch(
    img_u8: np.ndarray,
    gt: np.ndarray,
    prob: np.ndarray,
    out_path: str,
    thr: float = 0.5,
    text: str | None = None,
    max_side: int = 1600,
):
    pred = prob >= thr

    gt_bw = (gt.astype(np.uint8) * 255)
    pr_bw = (pred.astype(np.uint8) * 255)
    gt_bw = np.stack([gt_bw] * 3, axis=-1)
    pr_bw = np.stack([pr_bw] * 3, axis=-1)

    gt_ov = _overlay(img_u8, gt, color=(0, 255, 0), alpha=0.45)
    pr_ov = _overlay(img_u8, pred, color=(255, 0, 0), alpha=0.45)

    panels = [img_u8, gt_bw, pr_bw, gt_ov, pr_ov]
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


@dataclass
class _Rec:
    dice: float
    img_u8: np.ndarray
    gt: np.ndarray
    prob: np.ndarray
    slide_uid: str
    dataset: str
    crop_xy: Tuple[int, int]


class ValInspector:
    """
    Drop-in helper: minimal intrusion into your trainer.
    Call:
      insp = ValInspector(out_dir, epoch, k=20, thrs=[0.3,0.5,0.7], save_thr=0.5)
      insp.update(batch, prob_np, gt_np, dice)
      ...
      insp.finalize()
    """

    def __init__(
        self,
        out_dir: str,
        epoch: int,
        k: int = 20,
        thrs: List[float] = None,
        save_thr: float = 0.5,
        max_side: int = 1600,
    ):
        self.out_dir = str(out_dir)
        self.epoch = int(epoch)
        self.k = int(k)
        self.thrs = thrs if thrs is not None else [0.3, 0.5, 0.7]
        self.save_thr = float(save_thr)
        self.max_side = int(max_side)
        self._bank: List[_Rec] = []

    def update(self, batch: Dict[str, Any], prob_np: np.ndarray, gt_np: np.ndarray, dice: float):
        # assumes val batch_size=1
        img = batch["image"][0].detach().cpu().permute(1, 2, 0).numpy()
        img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)

        slide_uid = batch["slide_uid"][0]
        dataset = batch.get("dataset", ["unknown"])[0]

        cxy = batch.get("crop_xy", None)
        if cxy is None:
            crop_xy = (0, 0)
        else:
            # default collate makes tensor [1,2]
            try:
                cx, cy = cxy[0].tolist()
                crop_xy = (int(cx), int(cy))
            except Exception:
                crop_xy = tuple(map(int, cxy[0]))

        self._bank.append(
            _Rec(
                dice=float(dice),
                img_u8=img_u8,
                gt=gt_np.astype(bool),
                prob=prob_np.astype(np.float32),
                slide_uid=str(slide_uid),
                dataset=str(dataset),
                crop_xy=crop_xy,
            )
        )

    def finalize(self) -> Dict[str, float]:
        out = {}

        # threshold sweep on nonempty GTs
        for thr in self.thrs:
            ds = []
            for r in self._bank:
                if r.gt.sum() == 0:
                    continue
                ds.append(_dice_from_prob(r.prob, r.gt, thr))
            out[f"dice_nonempty_thr_{thr:.1f}"] = float(np.mean(ds)) if len(ds) else float("nan")

        # save worst-k
        if self.k > 0 and len(self._bank) > 0:
            self._bank.sort(key=lambda r: r.dice)  # ascending
            pick = self._bank[: min(self.k, len(self._bank))]
            out_dir = Path(self.out_dir) / "debug_vis_patches" / f"epoch{self.epoch:03d}" / "worst"
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, r in enumerate(pick):
                cx, cy = r.crop_xy
                name = f"{i:02d}_dice{r.dice:.3f}_{r.dataset}_{_sanitize(r.slide_uid)}_x{cx}_y{cy}.png"
                text = f"dice={r.dice:.3f} thr={self.save_thr:.2f} {r.dataset} {r.slide_uid} ({cx},{cy}) gt_fg={int(r.gt.sum())}"
                save_side_by_side_patch(
                    img_u8=r.img_u8,
                    gt=r.gt,
                    prob=r.prob,
                    out_path=str(out_dir / name),
                    thr=self.save_thr,
                    text=text,
                    max_side=self.max_side,
                )

        return out
