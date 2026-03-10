#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# ----------------------------
# MAT loader (v7 / v7.3)
# ----------------------------
def load_mat_inst_map(mat_path: Path) -> np.ndarray:
    """
    Returns inst_map as int32 HxW.
    Tries scipy.io.loadmat first; falls back to h5py for v7.3 mats.
    """
    try:
        import scipy.io as sio
        d = sio.loadmat(mat_path)
        # common keys: inst_map, instMap, instance_map
        for k in ["inst_map", "instMap", "instance_map", "inst_map_gt"]:
            if k in d:
                x = d[k]
                x = np.array(x)
                return x.astype(np.int32)
        raise KeyError(f"Cannot find inst_map in {mat_path}. Keys: {list(d.keys())[:30]}")
    except Exception as e:
        # v7.3 fallback (HDF5)
        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                f"Failed to load {mat_path} with scipy ({e}). "
                f"This looks like MATLAB v7.3; please `pip install h5py`."
            )
        with h5py.File(mat_path, "r") as f:
            # try common dataset names
            for k in ["inst_map", "instMap", "instance_map", "inst_map_gt"]:
                if k in f:
                    x = np.array(f[k])
                    # h5py often stores transposed (W,H) depending on writer;
                    # heuristic: choose orientation matching typical HxW (1000x1000)
                    if x.ndim == 2:
                        # if looks swapped, transpose
                        if x.shape[0] != x.shape[1] and x.shape[0] < x.shape[1]:
                            pass
                    return x.astype(np.int32)
            raise KeyError(f"Cannot find inst_map in v7.3 mat {mat_path}. Keys: {list(f.keys())[:30]}")

# ----------------------------
# COCO segm -> compressed RLE (bytes)
# ----------------------------
def segm_to_rle(segm: Any, h: int, w: int) -> Dict[str, Any]:
    """
    Convert polygon / uncompressed RLE (counts=list) / compressed RLE (counts=str/bytes)
    -> compressed RLE with counts=bytes for pycocotools.decode()
    """
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
        return rle

    if isinstance(segm, dict) and "counts" in segm:
        if isinstance(segm["counts"], list):
            # uncompressed RLE -> compressed
            rle = maskUtils.frPyObjects(segm, h, w)
            return rle
        rle = {"size": segm["size"], "counts": segm["counts"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle

    raise ValueError(f"Unknown segmentation format: {type(segm)}")

def decode_ann_mask(ann: dict, H: int, W: int) -> np.ndarray:
    rle = segm_to_rle(ann["segmentation"], H, W)
    m = maskUtils.decode(rle).astype(np.uint8)  # HxW
    return m

# ----------------------------
# quick visual helpers
# ----------------------------
def colorize_inst_map(inst_map: np.ndarray, seed: int = 0, alpha: float = 0.45) -> np.ndarray:
    """
    inst_map: HxW int32
    return: HxWx3 float32 overlay color (no base image)
    """
    H, W = inst_map.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    rng = np.random.default_rng(seed)

    # assign random colors per id (cheap for patch; ok for sanity)
    colors = rng.integers(0, 255, size=(len(ids), 3), dtype=np.int32).astype(np.float32)
    for c, iid in zip(colors, ids):
        m = (inst_map == iid)
        out[m] = (1 - alpha) * out[m] + alpha * c
    return out

def overlay_on_image(img: np.ndarray, overlay_rgb: np.ndarray) -> np.ndarray:
    """
    img: uint8 HxWx3
    overlay_rgb: float32 HxWx3 (already blended weight)
    """
    out = img.astype(np.float32).copy()
    m = overlay_rgb.sum(axis=2) > 0
    out[m] = np.clip(out[m] * 0.55 + overlay_rgb[m] * 1.0, 0, 255)
    return out.astype(np.uint8)

def xor_viz(coco_union: np.ndarray, mat_union: np.ndarray) -> np.ndarray:
    """
    red = coco only (FP), green = mat only (FN), yellow = both
    """
    H, W = coco_union.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    fp = (coco_union == 1) & (mat_union == 0)
    fn = (coco_union == 0) & (mat_union == 1)
    tp = (coco_union == 1) & (mat_union == 1)
    out[fp] = (255, 0, 0)
    out[fn] = (0, 255, 0)
    out[tp] = (255, 255, 0)
    return out

# ----------------------------
# metrics
# ----------------------------
def compute_union_stats(coco_union: np.ndarray, mat_union: np.ndarray) -> Dict[str, float]:
    inter = float(np.logical_and(coco_union, mat_union).sum())
    union = float(np.logical_or(coco_union, mat_union).sum())
    coco_area = float(coco_union.sum())
    mat_area = float(mat_union.sum())
    iou = inter / (union + 1e-9)
    prec = inter / (coco_area + 1e-9)   # pixel precision
    rec = inter / (mat_area + 1e-9)     # pixel recall
    return {"union_iou": iou, "pix_prec": prec, "pix_rec": rec, "coco_area": coco_area, "mat_area": mat_area}

def best_iou_per_coco_instance(
    coco_masks: List[np.ndarray],
    inst_map: np.ndarray,
) -> np.ndarray:
    """
    For each COCO instance mask, find best IoU w.r.t. MAT inst_map IDs.
    Efficient: use bincount over inst_map[mask].
    Returns array of best IoUs (len = #coco_instances).
    """
    flat = inst_map.ravel()
    max_id = int(flat.max())
    area_per_id = np.bincount(flat, minlength=max_id + 1).astype(np.float32)  # includes id=0 bg

    best = np.zeros((len(coco_masks),), dtype=np.float32)
    for i, m in enumerate(coco_masks):
        m = m.astype(bool)
        area_c = float(m.sum())
        if area_c <= 0:
            best[i] = 0.0
            continue
        ids = inst_map[m]
        inter = np.bincount(ids.ravel(), minlength=max_id + 1).astype(np.float32)
        # ignore background id=0
        cand = np.nonzero(inter[1:] > 0)[0] + 1
        if cand.size == 0:
            best[i] = 0.0
            continue
        inter_c = inter[cand]
        union_c = area_c + area_per_id[cand] - inter_c
        ious = inter_c / (union_c + 1e-9)
        best[i] = float(ious.max())
    return best

# ----------------------------
# patch filename parser
# ----------------------------
PATCH_RE = re.compile(r"(?P<srcid>\d+)__x(?P<x>\d+)_y(?P<y>\d+)", re.IGNORECASE)

def parse_patch_xy_from_name(file_name: str) -> Optional[Tuple[int, int, int]]:
    stem = Path(file_name).stem
    m = PATCH_RE.search(stem)
    if not m:
        return None
    srcid = int(m.group("srcid"))
    x0 = int(m.group("x"))
    y0 = int(m.group("y"))
    return srcid, x0, y0

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="COCO json (full or patch)")
    ap.add_argument("--img_root", required=True, help="image folder for that coco_json")
    ap.add_argument("--mat_root", required=True, help="folder containing .mat labels (CoNSeP Labels)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--num_images", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.45)

    ap.add_argument("--patch_mode", action="store_true",
                    help="If set, coco_json/img_root are patches; will crop MAT inst_map using x0/y0 from patch file name.")
    ap.add_argument("--src_coco_json", default=None,
                    help="If patch_mode, provide original(full-image) COCO json to map source image_id -> source file_name")
    ap.add_argument("--iou_thr1", type=float, default=0.50)
    ap.add_argument("--iou_thr2", type=float, default=0.75)

    args = ap.parse_args()

    coco = COCO(args.coco_json)
    img_root = Path(args.img_root)
    mat_root = Path(args.mat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_coco = None
    if args.patch_mode:
        if args.src_coco_json is None:
            raise ValueError("--patch_mode requires --src_coco_json to find the original file_name for .mat")
        src_coco = COCO(args.src_coco_json)

    ids = coco.getImgIds()
    random.seed(args.seed)
    random.shuffle(ids)
    ids = ids[: args.num_images]

    for img_id in ids:
        info = coco.loadImgs([img_id])[0]
        fn = info["file_name"]
        img_path = img_root / fn
        if not img_path.exists():
            print(f"[WARN] missing image: {img_path}")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # load coco anns & masks
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        coco_masks = []
        coco_union = np.zeros((H, W), dtype=np.uint8)
        for a in anns:
            if "segmentation" not in a:
                continue
            m = decode_ann_mask(a, H, W)
            coco_masks.append(m)
            coco_union |= (m > 0).astype(np.uint8)

        # locate MAT label and load inst_map
        if args.patch_mode:
            parsed = parse_patch_xy_from_name(fn)
            if parsed is None:
                print(f"[WARN] patch file_name not in expected format: {fn}")
                continue
            srcid, x0, y0 = parsed
            src_info = src_coco.loadImgs([srcid])[0]
            src_fn = src_info["file_name"]
            mat_path = mat_root / (Path(src_fn).stem + ".mat")
            inst_map_full = load_mat_inst_map(mat_path)
            inst_map = inst_map_full[y0:y0 + H, x0:x0 + W].astype(np.int32)
        else:
            mat_path = mat_root / (Path(fn).stem + ".mat")
            inst_map = load_mat_inst_map(mat_path).astype(np.int32)
            # some mats may load with extra dims
            if inst_map.ndim != 2:
                inst_map = np.squeeze(inst_map)
            if inst_map.shape[0] != H or inst_map.shape[1] != W:
                # last resort: try transpose
                if inst_map.T.shape == (H, W):
                    inst_map = inst_map.T
                else:
                    print(f"[WARN] mat inst_map shape {inst_map.shape} != image {(H,W)} for {fn}")
                    # still attempt crop to min size
                    hh = min(H, inst_map.shape[0]); ww = min(W, inst_map.shape[1])
                    inst_map = inst_map[:hh, :ww]
                    img = img[:hh, :ww]
                    H, W = img.shape[:2]
                    coco_union = coco_union[:hh, :ww]
                    coco_masks = [m[:hh, :ww] for m in coco_masks]

        mat_union = (inst_map > 0).astype(np.uint8)

        # metrics
        union_stats = compute_union_stats(coco_union, mat_union)
        best_ious = best_iou_per_coco_instance(coco_masks, inst_map) if len(coco_masks) > 0 else np.zeros((0,), np.float32)
        n_coco = len(coco_masks)
        n_mat = int(inst_map.max())

        frac_05 = float((best_ious >= args.iou_thr1).mean()) if n_coco > 0 else 0.0
        frac_075 = float((best_ious >= args.iou_thr2).mean()) if n_coco > 0 else 0.0

        print(
            f"[{img_id}] {fn} | MAT={mat_path.name} | "
            f"#coco={n_coco}, #mat={n_mat} | "
            f"unionIoU={union_stats['union_iou']:.3f} pixP={union_stats['pix_prec']:.3f} pixR={union_stats['pix_rec']:.3f} | "
            f"bestIoU>=0.5:{frac_05:.3f} >=0.75:{frac_075:.3f}"
        )

        # visuals
        coco_overlay = colorize_inst_map(
            # build a fake inst_map from coco masks (just for colors)
            # id-map is expensive; just colorize union by random per instance:
            # for visualization, we do simple per-mask blending
            inst_map=np.zeros((H, W), dtype=np.int32),
            seed=args.seed + img_id,
            alpha=args.alpha,
        )
        # draw coco by iterating masks:
        vis_coco = img.astype(np.float32).copy()
        rng = np.random.default_rng(args.seed + img_id)
        for m in coco_masks[:300]:
            m = m.astype(bool)
            if m.sum() == 0:
                continue
            c = rng.integers(0, 255, size=(3,), dtype=np.int32).astype(np.float32)
            vis_coco[m] = (1 - args.alpha) * vis_coco[m] + args.alpha * c
        vis_coco = np.clip(vis_coco, 0, 255).astype(np.uint8)

        mat_col = colorize_inst_map(inst_map, seed=args.seed + img_id, alpha=args.alpha)
        vis_mat = overlay_on_image(img, mat_col)

        diff = xor_viz(coco_union, mat_union)

        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(vis_coco); ax1.set_title(f"COCO overlay (n={n_coco})"); ax1.axis("off")
        ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(vis_mat);  ax2.set_title(f"MAT overlay (n={n_mat})");  ax2.axis("off")
        ax3 = fig.add_subplot(1, 3, 3); ax3.imshow(diff);     ax3.set_title("XOR: red=COCO only, green=MAT only"); ax3.axis("off")
        fig.tight_layout()

        out_path = out_dir / f"{Path(fn).stem}__c{n_coco}_m{n_mat}__IoU{union_stats['union_iou']:.3f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print("DONE.")

if __name__ == "__main__":
    main()