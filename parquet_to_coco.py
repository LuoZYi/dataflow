#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parquet_to_coco.py

Convert your pathology parquet (meta.parquet + ann_*.parquet) into a standard COCO JSON
that SAM3's COCO_FROM_JSON loader can read.

Key points:
- Output is standard COCO: {images, annotations, categories}
- segmentation is REQUIRED and is written as *uncompressed RLE* with counts=list[int]
- If your parquet stores bbox-cropped ROI RLE (rle_size_h/w = roi_h/w), we expand it to full-image RLE by default.

This works well with sam3.train.data.coco_json_loaders.COCO_FROM_JSON
(which auto-creates queries per category from COCO). See COCO_FROM_JSON.  (facebookresearch/sam3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------- RLE helpers (COCO-style, Fortran order) ----------------
def mask_to_rle_counts(mask: np.ndarray) -> List[int]:
    """
    COCO-style uncompressed RLE counts (Fortran order flatten).
    counts starts with number of 0s.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    pixels = mask.reshape(-1, order="F").astype(np.uint8)
    if pixels.size == 0:
        return [0]

    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    idx = np.concatenate(([0], changes, [pixels.size]))
    runs = np.diff(idx)

    # COCO expects counts start with zeros
    if pixels[0] == 1:
        runs = np.concatenate(([0], runs))

    return [int(x) for x in runs.tolist()]


def rle_counts_to_mask(counts: List[int], h: int, w: int) -> np.ndarray:
    """
    Decode uncompressed RLE counts (Fortran order) into a boolean mask of shape (h, w).
    counts starts with #zeros.
    """
    h = int(h)
    w = int(w)
    n = h * w
    arr = np.zeros(n, dtype=np.uint8)

    idx = 0
    val = 0  # 0 then 1 then 0...
    for run in counts:
        run = int(run)
        if run < 0:
            raise ValueError(f"Invalid RLE run < 0: {run}")
        if idx >= n:
            break
        end = idx + run
        if end > n:
            end = n
        if val == 1 and end > idx:
            arr[idx:end] = 1
        idx = end
        val ^= 1  # toggle

    return arr.reshape((h, w), order="F").astype(bool)


def expand_roi_rle_to_full(
    *,
    roi_counts: List[int],
    roi_h: int,
    roi_w: int,
    roi_x: int,
    roi_y: int,
    full_h: int,
    full_w: int,
) -> List[int]:
    """
    ROI RLE (defined on roi_h x roi_w) + top-left offset (roi_x, roi_y)
    -> full-image RLE counts on full_h x full_w.
    """
    roi_h = int(roi_h)
    roi_w = int(roi_w)
    roi_x = int(roi_x)
    roi_y = int(roi_y)
    full_h = int(full_h)
    full_w = int(full_w)

    roi_mask = rle_counts_to_mask(roi_counts, roi_h, roi_w)
    full = np.zeros((full_h, full_w), dtype=bool)

    # clip to image bounds (in case)
    x0 = max(0, roi_x)
    y0 = max(0, roi_y)
    x1 = min(full_w, roi_x + roi_w)
    y1 = min(full_h, roi_y + roi_h)

    if x1 <= x0 or y1 <= y0:
        # completely out of bounds -> empty
        return [full_h * full_w]

    rx0 = x0 - roi_x
    ry0 = y0 - roi_y
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    full[y0:y1, x0:x1] = roi_mask[ry0:ry1, rx0:rx1]
    return mask_to_rle_counts(full)


# ---------------- misc helpers ----------------
def _as_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def _norm_path_str(p: str) -> str:
    # Make JSON paths stable: forward slashes
    return _as_str(p).replace("\\", "/")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True, help="Path to meta.parquet")
    ap.add_argument("--ann", type=str, required=True, help="Path to ann_instance.parquet (or ann.parquet)")
    ap.add_argument("--img_root", type=str, required=True, help="Root folder containing image files")
    ap.add_argument("--out", type=str, required=True, help="Output COCO json path")

    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional: only keep meta rows with this split (e.g. train/val/test). "
             "If omitted -> use all splits.",
    )
    ap.add_argument(
        "--ann_kind",
        type=str,
        default="instance",
        choices=["instance", "semantic", "all"],
        help="Which annotations to export from ann parquet",
    )
    ap.add_argument(
        "--full_rle",
        type=int,
        default=1,
        help="1: expand ROI RLE to full-image RLE (RECOMMENDED for SAM3 COCO_FROM_JSON). "
             "0: assume rle_size == full image size already, otherwise error.",
    )
    ap.add_argument(
        "--drop_missing_files",
        type=int,
        default=1,
        help="1: drop images whose file does not exist under img_root. 0: do not check.",
    )
    ap.add_argument("--max_images", type=int, default=0, help="0=all, else keep first N images after filtering")

    return ap.parse_args()


def main():
    args = parse_args()
    meta_path = Path(args.meta)
    ann_path = Path(args.ann)
    img_root = Path(args.img_root)
    out_path = Path(args.out)

    assert meta_path.exists(), f"meta not found: {meta_path}"
    assert ann_path.exists(), f"ann not found: {ann_path}"
    assert img_root.exists(), f"img_root not found: {img_root}"

    # --- load meta ---
    meta_cols = ["slide_uid","rel_path","width_px","height_px","split","mpp_x","mpp_y"]
    meta = pd.read_parquet(meta_path, columns=[c for c in meta_cols if c in pd.read_parquet(meta_path, engine="pyarrow").columns])

    # Ensure required columns exist
    need = ["slide_uid", "rel_path", "width_px", "height_px"]
    for c in need:
        if c not in meta.columns:
            raise RuntimeError(f"meta.parquet missing required column: {c}. Have: {list(meta.columns)}")

    if args.split is not None:
        if "split" not in meta.columns:
            raise RuntimeError("You passed --split but meta.parquet has no 'split' column.")
        meta = meta[meta["split"].astype(str) == str(args.split)]

    meta = meta.copy()
    meta["rel_path"] = meta["rel_path"].astype(str).map(_norm_path_str)

    if args.drop_missing_files == 1:
        exists_mask = []
        for rp in meta["rel_path"].tolist():
            p = img_root / Path(rp)
            exists_mask.append(p.exists())
        exists_mask = np.asarray(exists_mask, dtype=bool)
        dropped = int((~exists_mask).sum())
        if dropped > 0:
            print(f"[WARN] drop_missing_files=1 -> dropping {dropped} images not found under {img_root}")
        meta = meta[exists_mask]

    if len(meta) == 0:
        raise RuntimeError(
            f"No meta rows after filtering. split={args.split}, drop_missing_files={args.drop_missing_files}.\n"
            f"Tip: check a few rel_path entries and whether (img_root/rel_path) exists."
        )

    if args.max_images and args.max_images > 0:
        meta = meta.head(args.max_images)

    # map slide_uid -> image_id
    slide_uids = meta["slide_uid"].astype(str).tolist()
    slide2imgid: Dict[str, int] = {su: i for i, su in enumerate(slide_uids)}

    images = []
    for su, rp, w, h, mx, my in zip(
        meta["slide_uid"].astype(str),
        meta["rel_path"].astype(str),
        meta["width_px"].astype(int),
        meta["height_px"].astype(int),
        meta["mpp_x"].fillna(-1).astype(float),
        meta["mpp_y"].fillna(-1).astype(float),
    ):
        images.append(
            {
                "id": int(slide2imgid[su]),
                "file_name": _norm_path_str(rp),
                "width": int(w),
                "height": int(h),
                "mpp_x": float(mx) if mx > 0 else -1.0,
                "mpp_y": float(my) if my > 0 else -1.0,
            }
        )
    # --- load annotations ---
    ann_need = [
        "slide_uid",
        "ann_kind",
        "label_id",
        "label_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
        "rle_size_h",
        "rle_size_w",
        "rle_counts",
        "area",
    ]
    ann_df = pd.read_parquet(ann_path)
    missing = [c for c in ann_need if c not in ann_df.columns]
    if missing:
        raise RuntimeError(f"ann parquet missing columns: {missing}. Have: {list(ann_df.columns)}")

    ann_df = ann_df.copy()
    ann_df["slide_uid"] = ann_df["slide_uid"].astype(str)

    # keep only metas we kept
    ann_df = ann_df[ann_df["slide_uid"].isin(set(slide2imgid.keys()))]

    # filter kind
    if args.ann_kind != "all":
        ann_df = ann_df[ann_df["ann_kind"].astype(str) == args.ann_kind]

    if len(ann_df) == 0:
        raise RuntimeError("No annotations left after filtering by meta + ann_kind. Check your ann parquet.")

    # categories: label_id -> label_name
    cat_map: Dict[int, str] = {}
    for lid, lname in zip(ann_df["label_id"].tolist(), ann_df["label_name"].tolist()):
        try:
            lid_i = int(lid)
        except Exception:
            continue
        if lid_i < 0:
            continue
        if lid_i not in cat_map:
            cat_map[lid_i] = str(lname)

    if len(cat_map) == 0:
        raise RuntimeError("No valid categories found (label_id all < 0?).")

    categories = [{"id": int(cid), "name": str(name)} for cid, name in sorted(cat_map.items(), key=lambda x: x[0])]

    # build COCO annotations
    annotations = []
    dropped_bad = 0
    for _, r in ann_df.iterrows():
        su = str(r["slide_uid"])
        if su not in slide2imgid:
            continue

        img_id = slide2imgid[su]
        full_w = int(meta.loc[meta["slide_uid"].astype(str) == su, "width_px"].iloc[0])
        full_h = int(meta.loc[meta["slide_uid"].astype(str) == su, "height_px"].iloc[0])

        # bbox in full-image coordinates (xywh)
        bx = int(r["bbox_x"])
        by = int(r["bbox_y"])
        bw = int(r["bbox_w"])
        bh = int(r["bbox_h"])
        if bw <= 0 or bh <= 0:
            dropped_bad += 1
            continue

        # category id
        try:
            cat_id = int(r["label_id"])
        except Exception:
            dropped_bad += 1
            continue
        if cat_id < 0:
            dropped_bad += 1
            continue

        # RLE
        roi_x = int(r["roi_x"])
        roi_y = int(r["roi_y"])
        roi_w = int(r["roi_w"])
        roi_h = int(r["roi_h"])
        rle_h = int(r["rle_size_h"])
        rle_w = int(r["rle_size_w"])

        counts_raw = r["rle_counts"]
        # counts could be list/np.ndarray/pyarrow list
        if isinstance(counts_raw, (np.ndarray,)):
            counts = [int(x) for x in counts_raw.tolist()]
        else:
            counts = [int(x) for x in list(counts_raw)]

        if args.full_rle == 1:
            # If already full-size + no offset, keep as is
            if rle_h == full_h and rle_w == full_w and roi_x == 0 and roi_y == 0:
                full_counts = counts
            else:
                # typical case: ROI/bbox RLE -> expand to full
                full_counts = expand_roi_rle_to_full(
                    roi_counts=counts,
                    roi_h=rle_h,
                    roi_w=rle_w,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    full_h=full_h,
                    full_w=full_w,
                )
            segmentation = {"size": [int(full_h), int(full_w)], "counts": full_counts}
        else:
            # only valid if your parquet already stores full-image RLE
            if not (rle_h == full_h and rle_w == full_w and roi_x == 0 and roi_y == 0):
                raise RuntimeError(
                    "full_rle=0 but annotation is not full-image RLE. "
                    "Set --full_rle 1."
                )
            segmentation = {"size": [int(full_h), int(full_w)], "counts": counts}

        area = r.get("area", None)
        try:
            area_i = int(area) if area is not None else int(bw * bh)
        except Exception:
            area_i = int(bw * bh)

        annotations.append(
            {
                "id": int(len(annotations)),
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(bx), float(by), float(bw), float(bh)],  # COCO xywh (absolute)
                "area": float(area_i),
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"images: {len(images)} | annotations: {len(annotations)} | categories: {len(categories)}")
    if dropped_bad:
        print(f"[WARN] dropped_bad_annotations: {dropped_bad}")
    print(f"NOTE: file_name is relative. Set Sam3ImageDataset.img_folder = img_root: {img_root}")


if __name__ == "__main__":
    main()