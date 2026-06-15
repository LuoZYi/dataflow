from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import random

import numpy as np
from PIL import Image


def bbox_from_bool(m: np.ndarray):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]


def get_adapter(dataset: str, root: Path, val_fold: int):
    if dataset == "sicapv2":
        from adapters.sicapv2 import SICAPv2Adapter
        return SICAPv2Adapter(root, split_mode="official", val_fold=val_fold, skip_unlisted=True)

    if dataset == "cocahis":
        from adapters.cocahis import CoCaHisAdapter
        return CoCaHisAdapter(root)

    if dataset == "bcss":
        from adapters.bcss import BCSSAdapter
        return BCSSAdapter(root)

    raise ValueError(f"Unknown dataset: {dataset}")


def save_overlay(img: np.ndarray, anns, out_path: Path):
    vis = img.copy().astype(np.uint8)

    colors = [
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 255, 0], dtype=np.uint8),
        np.array([0, 0, 255], dtype=np.uint8),
        np.array([255, 255, 0], dtype=np.uint8),
        np.array([255, 0, 255], dtype=np.uint8),
        np.array([0, 255, 255], dtype=np.uint8),
    ]

    for i, ann in enumerate(anns):
        m = ann.mask.astype(bool)
        color = colors[i % len(colors)]
        vis[m] = (0.55 * vis[m] + 0.45 * color).astype(np.uint8)

    Image.fromarray(vis).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sicapv2", "cocahis", "bcss"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--val-fold", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--num-overlays", type=int, default=50)
    ap.add_argument("--overlay-dir", default="/home/path_sam3/debug_adapter_audit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    ds = get_adapter(args.dataset, root, args.val_fold)

    samples = list(ds.iter_samples())
    print("=== BASIC ===")
    print("dataset:", args.dataset)
    print("root:", root)
    print("num samples:", len(samples))
    print("split counts:", Counter(s.split for s in samples))
    print("unique sample ids:", len(set(s.sample_id for s in samples)))

    if len(samples) != len(set(s.sample_id for s in samples)):
        dup = Counter(s.sample_id for s in samples)
        print("DUPLICATE sample ids:", [k for k, v in dup.items() if v > 1][:20])

    if hasattr(ds, "_last_split_debug"):
        print("split debug:", getattr(ds, "_last_split_debug"))

    n = min(args.max_samples, len(samples))
    chosen = samples[:n]

    ann_count = Counter()
    label_count = Counter()
    label_id_count = Counter()
    area_by_label = defaultdict(list)

    errors = []
    empty_samples = []

    print("\n=== CHECKING SAMPLE / ANN CONSISTENCY ===")

    for idx, s in enumerate(chosen):
        try:
            img = ds.load_image(s)
            anns = list(ds.iter_ann(s))
        except Exception as e:
            errors.append((s.sample_id, "load_or_iter_error", repr(e)))
            continue

        if img.ndim != 3 or img.shape[-1] != 3:
            errors.append((s.sample_id, "bad_image_shape", img.shape))
            continue

        H, W = img.shape[:2]

        meta_h = s.meta.get("height_px") if hasattr(s, "meta") else None
        meta_w = s.meta.get("width_px") if hasattr(s, "meta") else None
        if meta_h is not None and int(meta_h) != H:
            errors.append((s.sample_id, "height_meta_mismatch", (meta_h, H)))
        if meta_w is not None and int(meta_w) != W:
            errors.append((s.sample_id, "width_meta_mismatch", (meta_w, W)))

        ann_count[len(anns)] += 1
        if not anns:
            empty_samples.append(s.sample_id)

        for a in anns:
            m = np.asarray(a.mask).astype(bool)

            if m.shape != (H, W):
                errors.append((s.sample_id, a.ann_id, "mask_shape_mismatch", m.shape, (H, W)))
                continue

            calc_area = int(m.sum())
            if int(a.area) != calc_area:
                errors.append((s.sample_id, a.ann_id, "area_mismatch", int(a.area), calc_area))

            calc_bbox = bbox_from_bool(m)
            if list(a.bbox_xywh) != calc_bbox:
                errors.append((s.sample_id, a.ann_id, "bbox_mismatch", a.bbox_xywh, calc_bbox))

            if calc_area <= 0:
                errors.append((s.sample_id, a.ann_id, "zero_area"))

            x, y, bw, bh = calc_bbox
            if x < 0 or y < 0 or x + bw > W or y + bh > H:
                errors.append((s.sample_id, a.ann_id, "bbox_outside_image", calc_bbox, (H, W)))

            label_count[str(a.source_label)] += 1
            label_id_count[int(a.source_label_id)] += 1
            area_by_label[str(a.source_label)].append(calc_area)

        if idx % 500 == 0:
            print("checked", idx, "/", n)

    print("\n=== ANN SUMMARY ===")
    print("checked samples:", n)
    print("ann count per sample:", ann_count)
    print("label count:", label_count)
    print("label id count:", label_id_count)
    print("num empty samples:", len(empty_samples))
    print("first empty samples:", empty_samples[:20])

    print("\n=== AREA SUMMARY ===")
    for lab, areas in area_by_label.items():
        arr = np.asarray(areas)
        print(
            lab,
            "n=", len(arr),
            "min=", int(arr.min()),
            "p50=", int(np.percentile(arr, 50)),
            "p95=", int(np.percentile(arr, 95)),
            "max=", int(arr.max()),
        )

    print("\n=== ERRORS ===")
    print("num errors:", len(errors))
    for e in errors[:50]:
        print(e)

    out_dir = Path(args.overlay_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SAVING OVERLAYS ===")
    rng = random.Random(args.seed)
    overlay_candidates = [s for s in samples if len(list(ds.iter_ann(s))) > 0]
    rng.shuffle(overlay_candidates)

    for i, s in enumerate(overlay_candidates[: args.num_overlays]):
        img = ds.load_image(s)
        anns = list(ds.iter_ann(s))
        out_path = out_dir / f"{i:03d}_{s.sample_id}_{s.split}.jpg"
        save_overlay(img, anns, out_path)

    print("overlay dir:", out_dir)


if __name__ == "__main__":
    main()