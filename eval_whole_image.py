#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_whole_image.py

Whole-image evaluation for SAM3 on COCO instance segm with tiled/sliding-window inference.

Key features:
- Tile inference on original whole images (to bypass num_queries=200 density issue)
- "Ownership by center" to avoid duplicates from overlapping tiles
- Optional box-NMS as a second safety net
- Robust checkpoint loading:
  * can load a plain state_dict
  * can load a trainer checkpoint dict with ["model"]
  * auto-tries prefix fix: add "detector." if model expects it (common SAM3 mismatch)
- COCOeval with configurable maxDets (so you don't get stuck at maxDets=100)

Example:
python eval_whole_image.py \
  --coco_json /home/path_sam3/dataflow/generated_coco/consep/consep_test_instance.coco.json \
  --img_root  /home/path_sam3/pipeline/data_links/CoNSeP/CoNSeP \
  --base_ckpt /home/path_sam3/.hf/hub/models--facebook--sam3/snapshots/.../sam3.pt \
  --ft_ckpt   /home/path_sam3/dataflow/runs/consep/consep_patch/checkpoints/checkpoint_last.pt \
  --out_json  /home/path_sam3/dataflow/runs/consep/consep_patch/preds_whole.json \
  --tile 384 --overlap 96 --conf 0.05 --max_dets 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torchvision

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ----------------------- utils -----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def gen_windows(W: int, H: int, tile: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """
    Generate windows that cover the full image. Windows are (x0, y0, w, h).
    Ensure last window reaches the border.
    """
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]
    if xs[-1] != max(W - tile, 0):
        xs.append(max(W - tile, 0))
    if ys[-1] != max(H - tile, 0):
        ys.append(max(H - tile, 0))

    wins: List[Tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            pw = min(tile, W - x0)   # ✅ key fix
            ph = min(tile, H - y0)   # ✅ key fix
            wins.append((int(x0), int(y0), int(pw), int(ph)))
    return wins


def core_region(x0: int, y0: int, tile: int, stride: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """
    Define a non-overlapping ownership region ("core") for each tile.
    This partitions the full image into stride-sized cells (except borders).

    For a tile at (x0,y0), its core is:
      [x0, min(x0+stride, W)) x [y0, min(y0+stride, H))
    but if this tile is the last tile on that axis (x0 == W - tile), core extends to W.
    """
    # last-tile detection
    last_x0 = max(W - tile, 0)
    last_y0 = max(H - tile, 0)

    cx0 = x0
    cy0 = y0
    cx1 = W if x0 == last_x0 else min(x0 + stride, W)
    cy1 = H if y0 == last_y0 else min(y0 + stride, H)
    return cx0, cy0, cx1, cy1


def rle_encode_full_from_patch_mask(mask_patch, x0: int, y0: int, H: int, W: int) -> Dict[str, Any]:
    mask_patch = np.asarray(mask_patch)

    # squeeze possible channel dim
    if mask_patch.ndim == 3 and mask_patch.shape[0] == 1:
        mask_patch = mask_patch[0]
    elif mask_patch.ndim == 3 and mask_patch.shape[-1] == 1:
        mask_patch = mask_patch[..., 0]
    elif mask_patch.ndim != 2:
        raise RuntimeError(f"Unexpected single mask shape: {mask_patch.shape}")

    ph, pw = mask_patch.shape

    # clip to image border (important for last tiles)
    y1 = min(y0 + ph, H)
    x1 = min(x0 + pw, W)
    ph2, pw2 = y1 - y0, x1 - x0

    full = np.zeros((H, W), dtype=np.uint8, order="F")
    full[y0:y1, x0:x1] = mask_patch[:ph2, :pw2].astype(np.uint8)

    rle = maskUtils.encode(full)
    if isinstance(rle["counts"], (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def load_finetuned_state_dict(ft_ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a finetuned checkpoint that might be:
    - a plain state_dict (tensor values)
    - a dict with key 'model' storing state_dict
    - a dict with key 'state_dict'
    """
    ckpt = torch.load(ft_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            # maybe already a state_dict
            # keep only tensor values
            sd = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # strip DDP prefix if present
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def smart_load_finetune_into_model(model: torch.nn.Module, ft_ckpt_path: str) -> None:
    """
    Try to load finetuned weights into the built SAM3 image model.
    Automatically fixes the common 'detector.' prefix mismatch if needed.
    """
    sd = load_finetuned_state_dict(ft_ckpt_path)
    model_keys = set(model.state_dict().keys())

    direct = sum(1 for k in sd.keys() if k in model_keys)
    det_pref = sum(1 for k in sd.keys() if ("detector." + k) in model_keys)

    # Heuristic: if almost nothing matches directly but many would match with "detector." prefix,
    # remap keys by adding "detector.".
    if det_pref > direct * 5 and det_pref > 100:
        remap = {}
        for k, v in sd.items():
            k2 = "detector." + k
            if k2 in model_keys:
                remap[k2] = v
            elif k in model_keys:
                remap[k] = v
        sd = remap

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded finetune={ft_ckpt_path}")
    print(f"[ckpt] match_direct={direct} match_if_detector_prefix={det_pref}")
    print(f"[ckpt] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if len(missing) > 0:
        print(f"[ckpt] (first 20) missing: {missing[:20]}")
    if len(unexpected) > 0:
        print(f"[ckpt] (first 20) unexpected: {unexpected[:20]}")


# ----------------------- main -----------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="GT COCO json on WHOLE images")
    ap.add_argument("--img_root", required=True, help="Root folder such that img_root / file_name exists")
    ap.add_argument("--out_json", required=True, help="Where to save merged whole-image predictions json")

    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=96)
    ap.add_argument("--conf", type=float, default=0.05, help="SAM3 processor confidence threshold (try 0.05 first)")
    ap.add_argument("--max_dets", type=int, default=300, help="max detections per image for eval + truncation")

    ap.add_argument("--nms_iou", type=float, default=0.5, help="box NMS IoU (safety net); set <=0 to disable")

    # checkpoints
    ap.add_argument("--base_ckpt", default=None, help="Path to official sam3.pt (optional). If None, download from HF.")
    ap.add_argument("--ft_ckpt", required=True, help="Your finetuned checkpoint (trainer ckpt or state_dict).")
    ap.add_argument("--bpe_path", default=None, help="Optional BPE path; if None, use sam3 default asset")

    ap.add_argument("--limit", type=int, default=-1, help="debug: only eval first N images")
    args = ap.parse_args()

    stride = args.tile - args.overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")

    coco = COCO(args.coco_json)
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    cats = coco.loadCats(coco.getCatIds())
    # use category names as prompts (matches COCO_FROM_JSON behavior)
    cat_prompts = [(c["id"], c.get("name", str(c["id"]))) for c in cats]

    # ---- build model (load base first) ----
    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=True,
        checkpoint_path=args.base_ckpt,       # if None + load_from_HF=True => downloads
        load_from_HF=(args.base_ckpt is None),
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    # ---- load finetune weights (with auto prefix fix) ----
    smart_load_finetune_into_model(model, args.ft_ckpt)
    model.eval()

    processor = Sam3Processor(model)
    processor.set_confidence_threshold(args.conf)

    out_path = Path(args.out_json)
    ensure_dir(out_path.parent)

    preds: List[Dict[str, Any]] = []
    ann_id = 1

    for idx, img_id in enumerate(img_ids, 1):
        img = coco.loadImgs([img_id])[0]
        fn = img["file_name"]
        W, H = int(img["width"]), int(img["height"])
        img_path = Path(args.img_root) / fn
        if not img_path.exists():
            print(f"[WARN] missing image: {img_path}")
            continue

        im = Image.open(img_path).convert("RGB")
        windows = gen_windows(W, H, args.tile, stride)

        # collect candidates per category
        per_cat_boxes: Dict[int, List[List[float]]] = {cid: [] for cid, _ in cat_prompts}
        per_cat_scores: Dict[int, List[float]] = {cid: [] for cid, _ in cat_prompts}
        per_cat_masks: Dict[int, List[np.ndarray]] = {cid: [] for cid, _ in cat_prompts}
        per_cat_offsets: Dict[int, List[Tuple[int,int,int,int]]] = {cid: [] for cid, _ in cat_prompts}  # (x0,y0,pw,ph)

        for (x0, y0, pw, ph) in windows:
            patch = im.crop((x0, y0, x0 + pw, y0 + ph))
            state = processor.set_image(patch)

            # define ownership core for this tile
            cx0, cy0, cx1, cy1 = core_region(x0, y0, args.tile, stride, W, H)

            for (cat_id, prompt) in cat_prompts:
                out = processor.set_text_prompt(state=state, prompt=prompt)

                boxes = out.get("boxes", None)
                scores = out.get("scores", None)
                masks = out.get("masks", None)

                if boxes is None or scores is None or masks is None:
                    continue
                if len(scores) == 0:
                    continue

                # tensors -> cpu numpy
                boxes_np = boxes.detach().float().cpu().numpy()   # Nx4, xyxy in patch coords
                scores_np = scores.detach().float().cpu().numpy() # N
                # masks_np = masks.detach().cpu().numpy().astype(np.uint8)  # NxHpatchxWpatch
                masks_np = masks.detach().cpu().numpy()  # could be (N,1,H,W) or (N,H,W)
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0]  # -> (N,H,W)
                elif masks_np.ndim != 3:
                    raise RuntimeError(f"Unexpected masks shape: {masks_np.shape}")

                # 如果是 logits/prob，阈值一下；如果已经是 0/1，也没影响
                masks_np = (masks_np > 0.5).astype(np.uint8)

                # ownership filter: keep instance if global bbox center in this tile's core region
                for b, s, m in zip(boxes_np, scores_np, masks_np):
                    x1, y1, x2, y2 = b.tolist()
                    gx1 = x1 + x0
                    gy1 = y1 + y0
                    gx2 = x2 + x0
                    gy2 = y2 + y0
                    cx = 0.5 * (gx1 + gx2)
                    cy = 0.5 * (gy1 + gy2)

                    if not (cx0 <= cx < cx1 and cy0 <= cy < cy1):
                        continue

                    # clip global box
                    gx1 = float(max(0.0, min(gx1, W)))
                    gy1 = float(max(0.0, min(gy1, H)))
                    gx2 = float(max(0.0, min(gx2, W)))
                    gy2 = float(max(0.0, min(gy2, H)))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue

                    per_cat_boxes[cat_id].append([gx1, gy1, gx2, gy2])
                    per_cat_scores[cat_id].append(float(s))
                    per_cat_masks[cat_id].append(m)
                    per_cat_offsets[cat_id].append((x0, y0, pw, ph))

        # optional NMS per category
        for (cat_id, _) in cat_prompts:
            boxes = per_cat_boxes[cat_id]
            scores = per_cat_scores[cat_id]
            masks_list = per_cat_masks[cat_id]
            offsets = per_cat_offsets[cat_id]

            if len(scores) == 0:
                continue

            keep_idx = list(range(len(scores)))
            if args.nms_iou and args.nms_iou > 0:
                t_boxes = torch.tensor(boxes, dtype=torch.float32)
                t_scores = torch.tensor(scores, dtype=torch.float32)
                keep = torchvision.ops.nms(t_boxes, t_scores, args.nms_iou).cpu().numpy().tolist()
                keep_idx = keep

            # sort by score desc and truncate to max_dets
            keep_idx = sorted(keep_idx, key=lambda i: scores[i], reverse=True)[: args.max_dets]

            for i in keep_idx:
                gx1, gy1, gx2, gy2 = boxes[i]
                score = scores[i]
                mask_patch = masks_list[i]
                x0, y0, pw, ph = offsets[i]

                # encode full-image RLE
                rle = rle_encode_full_from_patch_mask(mask_patch, x0, y0, H, W)

                # COCO bbox expects xywh
                bbox_xywh = [gx1, gy1, max(0.0, gx2 - gx1), max(0.0, gy2 - gy1)]

                preds.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "bbox": bbox_xywh,
                    "score": float(score),
                })
                ann_id += 1

        if idx % 5 == 0 or idx == len(img_ids):
            print(f"[{idx}/{len(img_ids)}] done img_id={img_id}, preds_so_far={len(preds)}")

    # save predictions
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"Saved predictions: {out_path}  (#preds={len(preds)})")

    # graceful handling: empty predictions
    if len(preds) == 0:
        print(
            "\n[ERROR] #preds=0 so COCOeval cannot run.\n"
            "Most likely causes:\n"
            "  (1) finetuned checkpoint not loaded (common SAM3 'detector.' prefix mismatch)\n"
            "  (2) conf threshold too high (try --conf 0.01 or 0.0)\n"
            "  (3) prompt mismatch (print your COCO categories and try that exact name)\n"
        )
        return

    # COCOeval
    coco_dt = coco.loadRes(str(out_path))
    coco_eval = COCOeval(coco, coco_dt, iouType="segm")

    # IMPORTANT: set maxDets so summary prints maxDets=args.max_dets (not default 100)
    coco_eval.params.maxDets = [1, 10, args.max_dets]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()