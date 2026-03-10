#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_patch_coco.py

Create a patch-based COCO dataset from an instance-segmentation COCO json.

- Supports segmentation as polygon or RLE; internally converts to RLE.
- Crops masks to patch and shifts coords.
- Uses "core assignment" to avoid duplicating the same GT instance across overlapping patches.
- Saves:
    out_dir/
      images/
      annotations/instances_patch.json
      patch_map.jsonl

Example:
python make_patch_coco.py \
  --coco_json /home/path_sam3/dataflow/generated_coco/consep/consep_train_instance.coco.json \
  --img_root  /home/path_sam3/pipeline/data_links/CoNSeP/CoNSeP \
  --out_dir   /home/path_sam3/dataflow/generated_coco/consep_patches_train \
  --patch_size 384 --overlap 96 --min_area 20

If you train with FixedMPPRescale(target_mpp=0.5) but CoNSeP is ~0.25 mpp,
and you want "patch_size in target_mpp pixels", set:
  --patch_size_target 384 --overlap_target 96 --input_mpp 0.25 --target_mpp 0.5
which will crop 768x768 raw patches so that AFTER rescale they become 384x384.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def segm_to_rle(segm, h, w):
    """
    Convert polygon / uncompressed RLE / compressed RLE -> compressed RLE (counts=bytes)
    """
    # polygon
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
        return rle

    # RLE dict
    if isinstance(segm, dict) and "counts" in segm:
        # uncompressed RLE: counts is a list[int]
        if isinstance(segm["counts"], list):
            rle = maskUtils.frPyObjects(segm, h, w)  # returns compressed RLE
            return rle

        # compressed RLE: counts is str or bytes
        rle = {"size": segm["size"], "counts": segm["counts"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle

    raise ValueError(f"Unknown segmentation format: {type(segm)}")


def rle_to_bbox_area(rle: Dict[str, Any]) -> Tuple[List[float], float]:
    bb = maskUtils.toBbox(rle)  # [x,y,w,h]
    area = float(maskUtils.area(rle))
    bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    return bbox, area


def encode_mask(mask: np.ndarray) -> Dict[str, Any]:
    """mask: HxW uint8/bool; return json-serializable RLE (counts as str)."""
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle

def crop_rle(rle, x0, y0, pw, ph):
    """
    rle must be compressed with counts=bytes
    """
    # ensure counts bytes (extra safety)
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    elif isinstance(rle["counts"], list):
        # shouldn't happen if segm_to_rle is used, but keep safe
        rle = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])

    m = maskUtils.decode(rle).astype(np.uint8)  # HxW
    patch = m[y0:y0+ph, x0:x0+pw]
    if patch.sum() == 0:
        return None
    cropped = maskUtils.encode(np.asfortranarray(patch))
    if isinstance(cropped["counts"], bytes):
        cropped["counts"] = cropped["counts"].decode("ascii")  # json friendly
    return cropped


def gen_windows(W: int, H: int, patch: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """Generate windows covering full image with last aligned to border."""
    xs = list(range(0, max(1, W - patch + 1), stride))
    ys = list(range(0, max(1, H - patch + 1), stride))
    if xs[-1] != max(0, W - patch):
        xs.append(max(0, W - patch))
    if ys[-1] != max(0, H - patch):
        ys.append(max(0, H - patch))
    return [(x, y, patch, patch) for y in ys for x in xs]


def core_bounds(x0: int, y0: int, pw: int, ph: int, W: int, H: int, margin: int) -> Tuple[int, int, int, int]:
    """
    Core region for unique assignment:
    - interior patches: shrink by margin
    - boundary patches: extend to image boundary (no gap)
    """
    cx0 = x0 if x0 == 0 else x0 + margin
    cy0 = y0 if y0 == 0 else y0 + margin
    cx1 = x0 + pw if x0 + pw == W else x0 + pw - margin
    cy1 = y0 + ph if y0 + ph == H else y0 + ph - margin
    # clamp
    cx0 = max(0, min(W, cx0))
    cx1 = max(0, min(W, cx1))
    cy0 = max(0, min(H, cy0))
    cy1 = max(0, min(H, cy1))
    return cx0, cy0, cx1, cy1


def pick_patch_for_instance(
    bx: float, by: float, bw: float, bh: float,
    windows: List[Tuple[int, int, int, int]],
    W: int, H: int, margin: int
) -> int | None:
    """Assign GT to exactly one patch: bbox center must fall into that patch's core."""
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    for i, (x0, y0, pw, ph) in enumerate(windows):
        c0, r0, c1, r1 = core_bounds(x0, y0, pw, ph, W, H, margin)
        if (cx >= c0) and (cx < c1) and (cy >= r0) and (cy < r1):
            return i
    # fallback: if something weird, assign by plain containment
    for i, (x0, y0, pw, ph) in enumerate(windows):
        if (cx >= x0) and (cx < x0 + pw) and (cy >= y0) and (cy < y0 + ph):
            return i
    return None


def main():
    import argparse
    import json
    from pathlib import Path
    from typing import Any, Dict, List

    from PIL import Image
    from pycocotools.coco import COCO

    # 你文件里应该已经有这些函数：
    # - ensure_dir
    # - gen_windows
    # - segm_to_rle
    # - crop_rle
    # - rle_to_bbox_area

    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_dir", required=True)

    # mode A: raw patch size in input pixel space
    ap.add_argument("--patch_size", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=96)

    # mode B: patch size defined in target_mpp pixel space (optional)
    ap.add_argument("--patch_size_target", type=int, default=None)
    ap.add_argument("--overlap_target", type=int, default=None)
    ap.add_argument("--input_mpp", type=float, default=None)
    ap.add_argument("--target_mpp", type=float, default=None)

    ap.add_argument("--min_area", type=int, default=20, help="drop tiny cropped fragments")
    ap.add_argument("--keep_empty_patches", action="store_true", help="keep patches with 0 GT")
    ap.add_argument("--max_gt_per_patch", type=int, default=180, help="hard cap per patch")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_img = out_dir / "images"
    out_ann = out_dir / "annotations"
    ensure_dir(out_img)
    ensure_dir(out_ann)

    coco = COCO(args.coco_json)
    imgs = coco.loadImgs(coco.getImgIds())
    categories = coco.dataset["categories"]

    # decide actual raw patch size
    patch_size = args.patch_size
    overlap = args.overlap
    if args.patch_size_target is not None:
        if args.input_mpp is None or args.target_mpp is None:
            raise ValueError("If using --patch_size_target, you must provide --input_mpp and --target_mpp")
        scale = args.target_mpp / args.input_mpp
        patch_size = int(round(args.patch_size_target * scale))
        overlap = int(round((args.overlap_target or (args.patch_size_target // 4)) * scale))

    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch_size")

    new_images: List[Dict[str, Any]] = []
    new_anns: List[Dict[str, Any]] = []

    patch_map_path = out_dir / "patch_map.jsonl"
    f_map = open(patch_map_path, "w", encoding="utf-8")

    next_image_id = 1
    next_ann_id = 1

    for img in imgs:
        img_id = img["id"]
        fn = img["file_name"]
        W, H = int(img["width"]), int(img["height"])

        img_path = Path(args.img_root) / fn
        if not img_path.exists():
            # robust fallback: sometimes fn already includes subfolders
            alt = Path(args.img_root) / Path(fn).name
            if alt.exists():
                img_path = alt
            else:
                print(f"[WARN] missing image: {img_path} (also tried {alt})")
                continue

        im = Image.open(img_path).convert("RGB")

        windows = gen_windows(W, H, patch_size, stride)

        # ---- 1) create patch images + patch image records ----
        patch_infos: List[Dict[str, Any]] = []
        for (x0, y0, pw, ph) in windows:
            patch_name = f"{img_id}__x{int(x0):04d}_y{int(y0):04d}.png"
            patch_img = im.crop((x0, y0, x0 + pw, y0 + ph))
            patch_img.save(out_img / patch_name)

            pid = next_image_id
            next_image_id += 1

            new_images.append({
                "id": pid,
                "file_name": patch_name,
                "width": pw,
                "height": ph,
            })

            f_map.write(json.dumps({
                "patch_image_id": pid,
                "source_image_id": img_id,
                "file_name": patch_name,
                "x0": int(x0), "y0": int(y0), "w": int(pw), "h": int(ph),
                "source_w": W, "source_h": H,
            }, ensure_ascii=False) + "\n")

            patch_infos.append({"pid": pid, "x0": x0, "y0": y0, "pw": pw, "ph": ph})

        # ---- 2) load original annotations once, convert to RLE once ----
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        ann_rles = []
        for a in anns:
            if "segmentation" not in a:
                continue
            rle = segm_to_rle(a["segmentation"], H, W)
            ann_rles.append((a, rle))

        # ---- 3) for each patch: include ALL intersecting instances (duplicates allowed) ----
        for i, (x0, y0, pw, ph) in enumerate(windows):
            pid = patch_infos[i]["pid"]
            patch_kept = 0

            for a, rle in ann_rles:
                bx, by, bw, bh = a["bbox"]
                # quick bbox intersection test
                if bx + bw <= x0 or by + bh <= y0 or bx >= x0 + pw or by >= y0 + ph:
                    continue

                cropped = crop_rle(rle, x0, y0, pw, ph)
                if cropped is None:
                    continue

                bbox_p, area_p = rle_to_bbox_area(cropped)
                if area_p < args.min_area:
                    continue

                new_anns.append({
                    "id": next_ann_id,
                    "image_id": pid,
                    "category_id": a["category_id"],
                    "segmentation": cropped,
                    "bbox": bbox_p,
                    "area": area_p,
                    "iscrowd": 0,
                })
                next_ann_id += 1
                patch_kept += 1

                if patch_kept >= args.max_gt_per_patch:
                    break

            # if not keeping empty patches, we’ll drop them later by filtering new_images
            # (so nothing special needed here)

    f_map.close()

    # drop empty patches if requested
    if not args.keep_empty_patches:
        kept_img_ids = {ann["image_id"] for ann in new_anns}
        new_images = [im for im in new_images if im["id"] in kept_img_ids]

    out_coco = {"images": new_images, "annotations": new_anns, "categories": categories}
    out_json = out_ann / "instances_patch.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_coco, f, ensure_ascii=False)

    print("DONE")
    print(f"patch images: {out_img}")
    print(f"patch coco:   {out_json}")
    print(f"patch map:    {patch_map_path}")
    print(f"#patch images kept: {len(new_images)}")
    print(f"#patch anns:        {len(new_anns)}")


if __name__ == "__main__":
    main()