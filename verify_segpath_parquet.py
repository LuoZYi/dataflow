#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def rle_counts_to_mask(counts, h: int, w: int) -> np.ndarray:
    h, w = int(h), int(w)
    n = h * w
    arr = np.zeros(n, dtype=np.uint8)

    idx = 0
    val = 0
    for run in counts:
        run = int(run)
        if run < 0:
            raise ValueError(f"negative RLE run: {run}")
        end = min(idx + run, n)
        if val == 1 and end > idx:
            arr[idx:end] = 1
        idx = end
        val ^= 1

    return arr.reshape((h, w), order="F").astype(bool)


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))

    if arr.ndim == 3:
        if (
            arr.shape[-1] >= 3
            and np.all(arr[..., 0] == arr[..., 1])
            and np.all(arr[..., 0] == arr[..., 2])
        ):
            arr = arr[..., 0]
        else:
            arr = np.any(arr[..., :3] > 0, axis=-1).astype(np.uint8)

    return arr > 0


def image_to_mask_path(img_path: Path) -> Path:
    s = str(img_path)

    if "/images/" in s:
        s = s.replace("/images/", "/masks/")
    else:
        raise ValueError(f"Cannot infer mask path because '/images/' not in path: {img_path}")

    p = Path(s)
    stem = p.stem

    for suf in ["_HE", "-HE", "_he", "-he"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            return p.with_name(stem + "_mask" + p.suffix)

    return p.with_name(stem + "_mask" + p.suffix)


def bbox_from_mask(mask: np.ndarray):
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def save_overlay(
    img_path: Path,
    orig_mask: np.ndarray,
    decoded_full: np.ndarray,
    bbox,
    out_path: Path,
    label: str,
):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).copy()

    # red = decoded RLE
    red = decoded_full > 0
    overlay = img_np.copy()
    overlay[red, 0] = 255
    overlay[red, 1] = (overlay[red, 1] * 0.35).astype(np.uint8)
    overlay[red, 2] = (overlay[red, 2] * 0.35).astype(np.uint8)

    # green = original mask pixels that are not in decoded, should normally be none
    mismatch_orig_only = (orig_mask > 0) & (~decoded_full)
    overlay[mismatch_orig_only, 0] = 0
    overlay[mismatch_orig_only, 1] = 255
    overlay[mismatch_orig_only, 2] = 0

    out = Image.fromarray(overlay)
    draw = ImageDraw.Draw(out)

    x, y, w, h = [int(v) for v in bbox]
    draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0), width=3)
    draw.text((5, 5), label, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--sample", type=int, default=500, help="number of annotations to verify; 0 = full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="/home/path_sam3/dataflow/segpath_parquet_verify")
    ap.add_argument("--vis", type=int, default=30, help="number of overlay images to save")
    ap.add_argument("--progress_every", type=int, default=1000)
    args = ap.parse_args()

    meta_path = Path(args.meta)
    ann_path = Path(args.ann)
    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Reading parquet...")
    meta = pd.read_parquet(meta_path)
    ann = pd.read_parquet(ann_path)

    print("meta rows/images:", len(meta))
    print("ann rows:", len(ann))

    print("\nann_kind counts:")
    print(ann["ann_kind"].value_counts(dropna=False).to_string())

    print("\nlabel counts:")
    print(
        ann.groupby(["label_id", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("label_id")
        .to_string(index=False)
    )

    # Basic internal checks
    print("\n[2/5] Internal sanity checks...")
    errors = []

    if meta["slide_uid"].duplicated().any():
        n = int(meta["slide_uid"].duplicated().sum())
        errors.append(("META", f"duplicated slide_uid: {n}"))

    meta_by_slide = meta.set_index("slide_uid", drop=False)
    meta_slide_uids = set(meta["slide_uid"].astype(str).tolist())
    ann_slide_uids = set(ann["slide_uid"].astype(str).tolist())

    missing_meta = ann_slide_uids - meta_slide_uids
    if missing_meta:
        errors.append(("ANN", f"ann slide_uid missing in meta: {len(missing_meta)}"))

    if "ann_uid" in ann.columns and ann["ann_uid"].duplicated().any():
        n = int(ann["ann_uid"].duplicated().sum())
        errors.append(("ANN", f"duplicated ann_uid: {n}"))

    if not (ann["ann_kind"].astype(str) == "semantic").all():
        bad = int((ann["ann_kind"].astype(str) != "semantic").sum())
        errors.append(("ANN", f"non-semantic rows: {bad}"))

    # Select rows
    if args.sample and args.sample > 0 and args.sample < len(ann):
        print(f"\n[3/5] Sampling {args.sample} annotations for pixel verification...")
        ann_check = ann.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    else:
        print(f"\n[3/5] Full pixel verification over {len(ann)} annotations...")
        ann_check = ann.reset_index(drop=True)

    label_counter = Counter()
    area_values = []
    mismatch_count = 0
    vis_saved = 0

    error_rows = []

    for i, r in ann_check.iterrows():
        if args.progress_every and i % args.progress_every == 0:
            print(f"  checked {i}/{len(ann_check)}", flush=True)

        slide_uid = str(r["slide_uid"])
        if slide_uid not in meta_by_slide.index:
            error_rows.append([slide_uid, r.get("ann_uid", ""), "missing_meta", ""])
            continue

        m = meta_by_slide.loc[slide_uid]

        img_path = img_root / str(m["rel_path"])
        if not img_path.exists():
            error_rows.append([slide_uid, r.get("ann_uid", ""), "missing_image", str(img_path)])
            continue

        try:
            mask_path = image_to_mask_path(img_path)
        except Exception as e:
            error_rows.append([slide_uid, r.get("ann_uid", ""), "infer_mask_failed", str(e)])
            continue

        if not mask_path.exists():
            error_rows.append([slide_uid, r.get("ann_uid", ""), "missing_mask", str(mask_path)])
            continue

        try:
            img = Image.open(img_path)
            W_img, H_img = img.size
        except Exception as e:
            error_rows.append([slide_uid, r.get("ann_uid", ""), "image_open_failed", str(e)])
            continue

        W_meta = int(m["width_px"])
        H_meta = int(m["height_px"])

        if (W_img, H_img) != (W_meta, H_meta):
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "image_size_mismatch",
                f"meta={(W_meta,H_meta)} actual={(W_img,H_img)}",
            ])
            continue

        try:
            orig_mask = load_binary_mask(mask_path)
        except Exception as e:
            error_rows.append([slide_uid, r.get("ann_uid", ""), "mask_open_failed", str(e)])
            continue

        if orig_mask.shape != (H_meta, W_meta):
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "mask_size_mismatch",
                f"mask={orig_mask.shape} image={(H_meta,W_meta)}",
            ])
            continue

        bx, by, bw, bh = int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])
        rx, ry, rw, rh = int(r["roi_x"]), int(r["roi_y"]), int(r["roi_w"]), int(r["roi_h"])
        rle_h, rle_w = int(r["rle_size_h"]), int(r["rle_size_w"])

        for name, x, y, w, h in [
            ("bbox", bx, by, bw, bh),
            ("roi", rx, ry, rw, rh),
        ]:
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W_meta or y + h > H_meta:
                error_rows.append([
                    slide_uid,
                    r.get("ann_uid", ""),
                    f"{name}_out_of_bounds",
                    f"{name}={(x,y,w,h)} image={(W_meta,H_meta)}",
                ])
                continue

        if (rle_h, rle_w) != (rh, rw):
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "rle_size_roi_mismatch",
                f"rle={(rle_h,rle_w)} roi={(rh,rw)}",
            ])
            continue

        try:
            decoded_roi = rle_counts_to_mask(r["rle_counts"], rle_h, rle_w)
        except Exception as e:
            error_rows.append([slide_uid, r.get("ann_uid", ""), "rle_decode_failed", str(e)])
            continue

        orig_roi = orig_mask[ry : ry + rh, rx : rx + rw]

        if decoded_roi.shape != orig_roi.shape:
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "decoded_shape_mismatch",
                f"decoded={decoded_roi.shape} orig_roi={orig_roi.shape}",
            ])
            continue

        if not np.array_equal(decoded_roi, orig_roi):
            mismatch_count += 1
            diff = int(np.logical_xor(decoded_roi, orig_roi).sum())
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "rle_original_mask_mismatch",
                f"diff_pixels={diff}",
            ])
            continue

        decoded_area = int(decoded_roi.sum())
        row_area = int(r["area"])
        area_values.append(decoded_area)
        label_counter[str(r["label_name"])] += 1

        if decoded_area != row_area:
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "area_mismatch",
                f"row_area={row_area} decoded_area={decoded_area}",
            ])

        full_decoded = np.zeros_like(orig_mask, dtype=bool)
        full_decoded[ry : ry + rh, rx : rx + rw] = decoded_roi

        actual_bbox = bbox_from_mask(full_decoded)
        if actual_bbox is None:
            error_rows.append([slide_uid, r.get("ann_uid", ""), "decoded_empty", ""])
            continue

        if tuple(actual_bbox) != (bx, by, bw, bh):
            error_rows.append([
                slide_uid,
                r.get("ann_uid", ""),
                "bbox_mismatch",
                f"row={(bx,by,bw,bh)} decoded={actual_bbox}",
            ])

        if vis_saved < args.vis:
            label = f"{r['label_name']} area={decoded_area}"
            out_path = out_dir / "overlays" / f"verify_{vis_saved:03d}_{str(r['label_name'])}_{Path(str(m['rel_path'])).stem}.png"
            save_overlay(
                img_path=img_path,
                orig_mask=orig_mask,
                decoded_full=full_decoded,
                bbox=(bx, by, bw, bh),
                out_path=out_path,
                label=label,
            )
            vis_saved += 1

    print("\n[4/5] Writing error report...")
    error_csv = out_dir / "segpath_parquet_verify_errors.csv"
    with error_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_uid", "ann_uid", "error_type", "detail"])
        writer.writerows(error_rows)

    print("\n[5/5] Summary")
    print("=" * 100)
    print("checked annotations:", len(ann_check))
    print("pixel mismatch rows:", mismatch_count)
    print("error rows:", len(error_rows))
    print("visualizations saved:", vis_saved)
    print("error csv:", error_csv)
    print("overlay dir:", out_dir / "overlays")

    if area_values:
        print("decoded area min:", min(area_values))
        print("decoded area max:", max(area_values))
        print("decoded area mean:", sum(area_values) / len(area_values))

    print("\nchecked label counts:")
    for k, v in sorted(label_counter.items()):
        print(f"  {k}: {v}")

    if errors:
        print("\nInternal errors:")
        for e in errors:
            print(e)

    if len(error_rows) == 0 and not errors:
        print("\n[OK] Parquet verification passed.")
    else:
        print("\n[WARN] Verification found issues. Check CSV above.")


if __name__ == "__main__":
    main()
