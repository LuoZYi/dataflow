#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def segm_to_rle(segm: Any, h: int, w: int) -> Dict[str, Any]:
    """
    Convert polygon / uncompressed RLE / compressed RLE -> compressed RLE (counts=bytes)
    """
    # polygon
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
        return rle

    # dict RLE
    if isinstance(segm, dict) and "counts" in segm:
        # uncompressed RLE: counts is list[int]
        if isinstance(segm["counts"], list):
            rle = maskUtils.frPyObjects(segm, h, w)  # compressed
            return rle

        # compressed RLE: counts is str or bytes
        rle = {"size": segm["size"], "counts": segm["counts"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle

    raise ValueError(f"Unknown segmentation format: {type(segm)}")


def overlay_instances(
    img: np.ndarray,
    anns: List[dict],
    alpha: float = 0.45,
    draw_bbox: bool = True,
    max_instances: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """
    img: HxWx3 uint8
    anns: COCO annotations for one image
    returns: overlaid image uint8
    """
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    H, W = img.shape[:2]

    # shuffle so colors are mixed
    idxs = list(range(len(anns)))
    rng.shuffle(idxs)
    idxs = idxs[:max_instances]

    for k, i in enumerate(idxs):
        a = anns[i]
        segm = a.get("segmentation", None)
        if segm is None:
            continue

        rle = segm_to_rle(segm, H, W)
        m = maskUtils.decode(rle).astype(bool)  # HxW

        if m.sum() == 0:
            continue

        color = rng.integers(0, 255, size=(3,), dtype=np.int32)
        color = color.astype(np.float32)

        # alpha blend on mask pixels
        out[m] = (1 - alpha) * out[m] + alpha * color

        if draw_bbox and "bbox" in a:
            x, y, w, h = a["bbox"]
            x0, y0 = int(round(x)), int(round(y))
            x1, y1 = int(round(x + w)), int(round(y + h))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W - 1, x1), min(H - 1, y1)

            # draw rectangle border (cheap)
            border = 2
            out[y0:y0+border, x0:x1] = color
            out[y1-border:y1, x0:x1] = color
            out[y0:y1, x0:x0+border] = color
            out[y0:y1, x1-border:x1] = color

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_images", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--max_instances", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw_bbox", action="store_true")
    args = ap.parse_args()

    coco = COCO(args.coco_json)
    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()
    random.seed(args.seed)
    random.shuffle(img_ids)
    img_ids = img_ids[: args.num_images]

    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        fn = img_info["file_name"]
        path = img_root / fn
        if not path.exists():
            print(f"[WARN] missing image: {path}")
            continue

        im = np.array(Image.open(path).convert("RGB"))
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        over = overlay_instances(
            im, anns,
            alpha=args.alpha,
            draw_bbox=args.draw_bbox,
            max_instances=args.max_instances,
            seed=args.seed + img_id,
        )

        # save
        out_path = out_dir / f"{Path(fn).stem}__n{len(anns)}.png"
        Image.fromarray(over).save(out_path)
        print(f"[OK] {img_id} {fn} anns={len(anns)} -> {out_path}")

    print("DONE.")


if __name__ == "__main__":
    main()