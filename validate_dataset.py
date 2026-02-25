#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_dataset.py

Validate ROI dataset adapters WITHOUT producing any intermediate files.
It reads samples via adapters, checks images & annotations consistency, and prints a summary.

Example:
  python validate_dataset.py --dataset glas --root /path/to/GlaS --max_samples 50
  python validate_dataset.py --dataset consep --root /path/to/CoNSeP
  python validate_dataset.py --dataset crag --root /path/to/CRAG --include_aug 0 --max_samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---- helpers (no dependency on your core utils; keep script standalone) ----
def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _area_from_mask(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _coerce_bool_mask(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        return None
    if mask.ndim != 2:
        return None
    if mask.dtype == np.bool_:
        return mask
    # common: uint8 0/1
    if np.issubdtype(mask.dtype, np.integer):
        return mask.astype(bool)
    return None


def _polygons_oob(polys: List[List[float]], h: int, w: int, eps: float = 1.0) -> int:
    """
    Count number of polygon points outside [-eps, w-1+eps] x [-eps, h-1+eps].
    """
    oob = 0
    for poly in polys:
        if not poly:
            continue
        if len(poly) % 2 != 0:
            # malformed poly
            oob += 1
            continue
        for i in range(0, len(poly), 2):
            x = float(poly[i])
            y = float(poly[i + 1])
            if x < -eps or x > (w - 1 + eps) or y < -eps or y > (h - 1 + eps):
                oob += 1
    return oob


# ---- adapter factory ----
def make_adapter(dataset: str, root: Path, args: argparse.Namespace):
    # Import here so script still runs even if some deps are missing for other datasets.
    from dataflow.adapters import (
        ConSepAdapter,
        CRAGAdapter,
        BCSSAdapter,
        GlaSAdapter,
        LizardAdapter,
        PanNukeAdapter,
    )

    dataset = dataset.lower()
    if dataset == "consep":
        return ConSepAdapter(root)
    if dataset == "crag":
        return CRAGAdapter(root, include_aug=bool(args.include_aug))
    if dataset == "bcss":
        return BCSSAdapter(root, split_mode=args.bcss_split_mode, seed=args.seed,
                           val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    if dataset == "glas":
        return GlaSAdapter(root)
    if dataset == "lizard":
        return LizardAdapter(root)
    if dataset == "pannuke":
        return PanNukeAdapter(root)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---- main validate ----
def validate(adapter, *, max_samples: int, fail_fast: bool, verbose: bool) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    split_counter = Counter()
    label_counter = Counter()
    kind_counter = Counter()

    n_samples = 0
    n_images_read_fail = 0
    n_ann_read_fail = 0

    n_empty_masks = 0
    n_shape_mismatch = 0
    n_bad_mask_type = 0
    n_bbox_mismatch = 0
    n_area_mismatch = 0
    n_poly_oob_points = 0

    errors: List[str] = []
    warns: List[str] = []

    def _err(msg: str):
        errors.append(msg)
        if fail_fast:
            raise RuntimeError(msg)

    def _warn(msg: str):
        warns.append(msg)

    for sample in adapter.iter_samples():
        if max_samples > 0 and n_samples >= max_samples:
            break

        n_samples += 1
        split_counter[str(sample.split)] += 1

        # ---- load image ----
        try:
            img = adapter.load_image(sample)
            if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                _warn(f"[{sample.sample_id}] image is not HxWx3 ndarray, got shape={getattr(img,'shape',None)}")
            h, w = int(img.shape[0]), int(img.shape[1])
        except Exception as e:
            n_images_read_fail += 1
            _err(f"[{sample.sample_id}] failed to read image: {sample.image_path} err={repr(e)}")
            continue

        # ---- load ann ----
        try:
            ann_list = list(adapter.iter_ann(sample))
        except Exception as e:
            n_ann_read_fail += 1
            _err(f"[{sample.sample_id}] failed to read annotations: {sample.ann_path} err={repr(e)}")
            continue

        if len(ann_list) == 0:
            _warn(f"[{sample.sample_id}] no annotations returned")
            continue

        for ann in ann_list:
            kind_counter[str(ann.kind)] += 1
            label_counter[str(ann.source_label)] += 1

            # polygons OOB check (if present)
            if ann.polygons is not None:
                try:
                    n_poly_oob_points += _polygons_oob(ann.polygons, h=h, w=w)
                except Exception:
                    _warn(f"[{sample.sample_id}] polygon check failed for ann_id={ann.ann_id}")

            mask = _coerce_bool_mask(ann.mask)

            if mask is None:
                # It's OK if polygon exists and you don't store mask; but for MVP we strongly prefer mask.
                if ann.polygons is None:
                    n_bad_mask_type += 1
                    _warn(f"[{sample.sample_id}] ann_id={ann.ann_id} has neither valid mask nor polygons")
                else:
                    _warn(f"[{sample.sample_id}] ann_id={ann.ann_id} has polygons but no valid mask (OK if intended)")
                continue

            if mask.shape != (h, w):
                n_shape_mismatch += 1
                _warn(f"[{sample.sample_id}] ann_id={ann.ann_id} mask shape mismatch: {mask.shape} vs image {(h,w)}")
                continue

            if not np.any(mask):
                n_empty_masks += 1
                _warn(f"[{sample.sample_id}] ann_id={ann.ann_id} empty mask")

            # bbox/area consistency
            bb = _bbox_from_mask(mask)
            ar = _area_from_mask(mask)

            if ann.bbox_xywh is not None and bb is not None:
                if tuple(map(int, ann.bbox_xywh)) != tuple(map(int, bb)):
                    n_bbox_mismatch += 1
                    if verbose:
                        _warn(f"[{sample.sample_id}] ann_id={ann.ann_id} bbox mismatch: "
                              f"adapter={ann.bbox_xywh} recompute={bb}")

            if ann.area is not None:
                if int(ann.area) != int(ar):
                    n_area_mismatch += 1
                    if verbose:
                        _warn(f"[{sample.sample_id}] ann_id={ann.ann_id} area mismatch: "
                              f"adapter={ann.area} recompute={ar}")

    stats["n_samples_checked"] = n_samples
    stats["split_counts"] = dict(split_counter)
    stats["ann_kind_counts"] = dict(kind_counter)
    stats["top_labels"] = label_counter.most_common(20)

    stats["failures"] = {
        "image_read_fail": n_images_read_fail,
        "ann_read_fail": n_ann_read_fail,
    }

    stats["warnings_counters"] = {
        "empty_masks": n_empty_masks,
        "mask_shape_mismatch": n_shape_mismatch,
        "bad_mask_type": n_bad_mask_type,
        "bbox_mismatch": n_bbox_mismatch,
        "area_mismatch": n_area_mismatch,
        "polygon_oob_points": n_poly_oob_points,
    }

    stats["errors"] = errors
    stats["warnings"] = warns[:200]  # avoid flooding
    stats["warnings_truncated"] = max(0, len(warns) - len(stats["warnings"]))

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["consep", "crag", "bcss", "glas", "lizard", "pannuke"])
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means check all")
    ap.add_argument("--fail_fast", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dump_json", type=str, default=None, help="Optional: dump report json to this path")

    # some dataset-specific knobs (still no intermediate files)
    ap.add_argument("--include_aug", type=int, default=1, help="CRAG: include augmented images (1/0)")
    ap.add_argument("--bcss_split_mode", choices=["unspecified", "random"], default="unspecified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        sys.exit(2)

    try:
        adapter = make_adapter(args.dataset, root, args)
    except Exception as e:
        print(f"[ERROR] failed to create adapter: {repr(e)}", file=sys.stderr)
        sys.exit(2)

    try:
        report = validate(
            adapter,
            max_samples=args.max_samples,
            fail_fast=args.fail_fast,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"[ERROR] validation crashed: {repr(e)}", file=sys.stderr)
        sys.exit(2)

    # Pretty print summary
    print("\n========== VALIDATION SUMMARY ==========")
    print(f"dataset: {args.dataset}")
    print(f"root:    {root}")
    print(f"samples_checked: {report['n_samples_checked']}")
    print(f"split_counts:    {report['split_counts']}")
    print(f"ann_kind_counts: {report['ann_kind_counts']}")
    print("top_labels:")
    for k, v in report["top_labels"]:
        print(f"  - {k}: {v}")

    print("\nfailures:", report["failures"])
    print("warnings_counters:", report["warnings_counters"])
    if report["errors"]:
        print(f"\n[ERRORS] ({len(report['errors'])}) show first 20:")
        for x in report["errors"][:20]:
            print("  ", x)

    if report["warnings"]:
        print(f"\n[WARNINGS] ({len(report['warnings'])}+{report['warnings_truncated']} truncated) show first 30:")
        for x in report["warnings"][:30]:
            print("  ", x)

    if args.dump_json:
        out = Path(args.dump_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] dumped json report: {out}")

    # exit code policy: if hard failures exist, nonzero
    hard_fail = report["failures"]["image_read_fail"] + report["failures"]["ann_read_fail"]
    if hard_fail > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
