#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_parquet.py

Generate meta/ann parquet files from multiple pathology datasets via adapters.
Supports:
- Fast H/W from sample.meta (height_px/width_px) or header-only read_hw_fast fallback
- RLE encoding (full-image or bbox ROI)
- Optional derived semantic masks by unioning instance masks per sample
- Output layout: merged (one set of parquets) or per_dataset (one folder per dataset)
- ann output mode: mixed (one ann.parquet) or split (ann_instance.parquet + ann_semantic.parquet)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- make imports robust (so you can run from anywhere) ---
_THIS = Path(__file__).resolve()
# assumes repo root is /home/path_sam3 and this file is under it (or any subfolder)
sys.path.insert(0, str(_THIS.parents[1]))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow is required to write parquet. Try: pip install pyarrow") from e


# ---------------- RLE (COCO-style) ----------------
def mask_to_rle_counts(mask: np.ndarray) -> List[int]:
    """
    COCO-style RLE counts, Fortran order flatten.
    counts starts with number of 0s.
    Vectorized-ish implementation.
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


def _bbox_from_mask_np(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Fast bbox from boolean mask. Returns (x, y, w, h) or None if empty."""
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y0 = int(np.argmax(rows))
    y1 = int(mask.shape[0] - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(mask.shape[1] - np.argmax(cols[::-1]))

    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    if w <= 0 or h <= 0:
        return None
    return x0, y0, w, h


# ---------------- adapter factory ----------------
def make_adapter(dataset: str, root: Path, args: argparse.Namespace):
    from dataflow.adapters import (
        ConSepAdapter,
        CRAGAdapter,
        BCSSAdapter,
        GlaSAdapter,
        LizardAdapter,
        PanNukeAdapter,
    )

    d = dataset.lower()
    if d == "glas":
        return GlaSAdapter(root)
    if d == "consep":
        return ConSepAdapter(root)
    if d == "crag":
        return CRAGAdapter(root, include_aug=bool(args.include_aug))
    if d == "bcss":
        return BCSSAdapter(
            root,
            split_mode=args.bcss_split_mode,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    if d == "lizard":
        return LizardAdapter(root)
    if d == "pannuke":
        return PanNukeAdapter(root)
    raise ValueError(f"Unknown dataset: {dataset}")


def parse_inputs(input_args: List[str]) -> List[Tuple[str, Path]]:
    """Parse repeated --input dataset=/path/to/root"""
    out: List[Tuple[str, Path]] = []
    for s in input_args:
        if "=" not in s:
            raise ValueError(f"--input expects dataset=/path, got: {s}")
        ds, rp = s.split("=", 1)
        out.append((ds.strip(), Path(rp).expanduser().resolve()))
    return out


# ---------------- parquet schemas ----------------
META_SCHEMA = pa.schema(
    [
        ("slide_uid", pa.string()),
        ("dataset", pa.string()),
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("group_id", pa.string()),
        ("rel_path", pa.string()),
        ("backend_type", pa.string()),
        ("width_px", pa.int32()),
        ("height_px", pa.int32()),
        ("mpp_x", pa.float32()),
        ("mpp_y", pa.float32()),
        ("tissue_type", pa.string()),
        ("stain_type", pa.string()),
        ("source", pa.string()),
        ("patient_id", pa.string()),
        ("extra", pa.string()),
    ]
)

ANN_SCHEMA = pa.schema(
    [
        ("ann_uid", pa.string()),
        ("slide_uid", pa.string()),
        ("dataset", pa.string()),
        ("task_type", pa.string()),      # "SEG"
        ("ann_kind", pa.string()),       # "instance"/"semantic"
        ("label_name", pa.string()),
        ("label_id", pa.int32()),
        ("roi_x", pa.int32()),
        ("roi_y", pa.int32()),
        ("roi_w", pa.int32()),
        ("roi_h", pa.int32()),
        ("geometry_fmt", pa.string()),   # "RLE"
        ("rle_order", pa.string()),      # "F"
        ("rle_size_h", pa.int32()),
        ("rle_size_w", pa.int32()),
        ("rle_counts", pa.list_(pa.int32())),
        ("bbox_x", pa.int32()),
        ("bbox_y", pa.int32()),
        ("bbox_w", pa.int32()),
        ("bbox_h", pa.int32()),
        ("area", pa.int32()),
        ("is_gt", pa.bool_()),
        ("confidence", pa.float32()),
        ("annotator", pa.string()),
        ("extra", pa.string()),
    ]
)


class ParquetBatchWriter:
    def __init__(self, out_path: Path, schema: pa.Schema, compression: str = "zstd"):
        self.out_path = out_path
        self.schema = schema
        self.compression = compression
        self._writer: Optional[pq.ParquetWriter] = None

    def write_pylist(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self._writer is None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                str(self.out_path),
                table.schema,
                compression=self.compression,
            )
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def _valid_mpp(v) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if x > 0 else None


def _get_hw_from_meta(sample) -> Tuple[Optional[int], Optional[int]]:
    if not sample.meta:
        return None, None
    h = sample.meta.get("height_px", None)
    w = sample.meta.get("width_px", None)
    try:
        h = int(h) if h is not None else None
        w = int(w) if w is not None else None
    except Exception:
        return None, None
    if h is not None and h > 0 and w is not None and w > 0:
        return h, w
    return None, None


def _clip_bbox_xywh(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(int(x), W))
    y0 = max(0, min(int(y), H))
    x1 = max(0, min(x0 + int(w), W))
    y1 = max(0, min(y0 + int(h), H))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _safe_label(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)[:80]


class _UnionMaskBuilder:
    """
    Maintain a union mask in a dynamically-growing ROI.
    Stores:
      roi_x, roi_y, mask (roi_h, roi_w) boolean.
    """

    def __init__(self, x: int, y: int, crop: np.ndarray):
        self.roi_x = int(x)
        self.roi_y = int(y)
        self.mask = crop.astype(bool).copy()

    @property
    def roi_h(self) -> int:
        return int(self.mask.shape[0])

    @property
    def roi_w(self) -> int:
        return int(self.mask.shape[1])

    def update(self, x: int, y: int, crop: np.ndarray):
        crop = crop.astype(bool)
        if crop.size == 0 or (not crop.any()):
            return

        x = int(x)
        y = int(y)
        h = int(crop.shape[0])
        w = int(crop.shape[1])
        if h <= 0 or w <= 0:
            return

        old_x0, old_y0 = self.roi_x, self.roi_y
        old_x1, old_y1 = old_x0 + self.roi_w, old_y0 + self.roi_h

        new_x0 = min(old_x0, x)
        new_y0 = min(old_y0, y)
        new_x1 = max(old_x1, x + w)
        new_y1 = max(old_y1, y + h)

        if new_x0 != old_x0 or new_y0 != old_y0 or new_x1 != old_x1 or new_y1 != old_y1:
            new_w = new_x1 - new_x0
            new_h = new_y1 - new_y0
            new_mask = np.zeros((new_h, new_w), dtype=bool)

            ox = old_x0 - new_x0
            oy = old_y0 - new_y0
            new_mask[oy : oy + self.roi_h, ox : ox + self.roi_w] = self.mask

            self.roi_x, self.roi_y = new_x0, new_y0
            self.mask = new_mask

        dx = x - self.roi_x
        dy = y - self.roi_y
        self.mask[dy : dy + h, dx : dx + w] |= crop


@dataclass
class _DatasetWriters:
    meta_writer: ParquetBatchWriter
    ann_writer: Optional[ParquetBatchWriter]
    ann_inst_writer: Optional[ParquetBatchWriter]
    ann_sem_writer: Optional[ParquetBatchWriter]
    meta_rows: List[Dict[str, Any]]
    ann_rows: List[Dict[str, Any]]
    ann_inst_rows: List[Dict[str, Any]]
    ann_sem_rows: List[Dict[str, Any]]
    n_samples: int = 0
    n_anns: int = 0

    def emit_ann_row(self, row: Dict[str, Any], args: argparse.Namespace):
        if args.ann_out_mode == "mixed":
            self.ann_rows.append(row)
            if len(self.ann_rows) >= args.chunk_size:
                assert self.ann_writer is not None
                self.ann_writer.write_pylist(self.ann_rows)
                self.ann_rows.clear()
        else:
            if row["ann_kind"] == "semantic":
                self.ann_sem_rows.append(row)
                if len(self.ann_sem_rows) >= args.chunk_size:
                    assert self.ann_sem_writer is not None
                    self.ann_sem_writer.write_pylist(self.ann_sem_rows)
                    self.ann_sem_rows.clear()
            else:
                self.ann_inst_rows.append(row)
                if len(self.ann_inst_rows) >= args.chunk_size:
                    assert self.ann_inst_writer is not None
                    self.ann_inst_writer.write_pylist(self.ann_inst_rows)
                    self.ann_inst_rows.clear()

        self.n_anns += 1

    def flush_all(self, args: argparse.Namespace):
        self.meta_writer.write_pylist(self.meta_rows)
        self.meta_rows.clear()

        if args.ann_out_mode == "mixed":
            assert self.ann_writer is not None
            self.ann_writer.write_pylist(self.ann_rows)
            self.ann_rows.clear()
        else:
            assert self.ann_inst_writer is not None and self.ann_sem_writer is not None
            self.ann_inst_writer.write_pylist(self.ann_inst_rows)
            self.ann_sem_writer.write_pylist(self.ann_sem_rows)
            self.ann_inst_rows.clear()
            self.ann_sem_rows.clear()

    def close(self, args: argparse.Namespace):
        self.flush_all(args)
        self.meta_writer.close()
        if args.ann_out_mode == "mixed":
            assert self.ann_writer is not None
            self.ann_writer.close()
        else:
            assert self.ann_inst_writer is not None and self.ann_sem_writer is not None
            self.ann_inst_writer.close()
            self.ann_sem_writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # inputs
    ap.add_argument("--input", action="append", default=[], help="dataset=/abs/path/to/root (repeatable)")
    ap.add_argument("--dataset", type=str, default=None, help="Single dataset name (if no --input)")
    ap.add_argument("--root", type=str, default=None, help="Single dataset root (if no --input)")

    # output layout
    ap.add_argument(
        "--out_layout",
        choices=["per_dataset", "merged"],
        default="per_dataset",
        help="per_dataset: out_dir/<dataset>/meta.parquet etc; merged: out_dir/meta.parquet etc",
    )

    # limits / performance
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    ap.add_argument("--chunk_size", type=int, default=2000, help="Rows per parquet write chunk")
    ap.add_argument("--compression", type=str, default="zstd", help="Parquet compression (zstd/snappy/gzip)")

    # dataset knobs
    ap.add_argument("--include_aug", type=int, default=1, help="CRAG: include augmented images (1/0)")
    ap.add_argument("--bcss_split_mode", choices=["unspecified", "random"], default="unspecified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    # mpp defaults (used only if adapter doesn't provide)
    ap.add_argument("--mpp", type=float, default=None, help="Default mpp (um/px). Sets both mpp_x and mpp_y.")
    ap.add_argument("--mpp_x", type=float, default=None, help="Default mpp_x (um/px).")
    ap.add_argument("--mpp_y", type=float, default=None, help="Default mpp_y (um/px).")

    # RLE ROI mode
    ap.add_argument(
        "--rle_roi",
        choices=["full", "bbox"],
        default="bbox",
        help="Encode mask as RLE over full image or bbox-cropped ROI (bbox is MUCH smaller).",
    )

    # store extra meta
    ap.add_argument(
        "--store_extra",
        action="store_true",
        help="If set, dump sample/ann meta into extra JSON string fields (bigger parquet).",
    )

    # derived semantic
    ap.add_argument(
        "--emit_semantic_from_instance",
        type=int,
        default=0,
        help="If 1, also emit ann_kind=semantic by unioning instance masks per sample.",
    )
    ap.add_argument(
        "--semantic_mode",
        choices=["per_label", "all"],
        default="per_label",
        help="per_label: union per (label_id,label_name); all: union all instances into one semantic.",
    )
    ap.add_argument(
        "--semantic_min_area",
        type=int,
        default=1,
        help="Skip derived semantic masks with area < this threshold.",
    )

    ap.add_argument(
        "--ann_out_mode",
        choices=["mixed", "split"],
        default="mixed",
        help="mixed: ann.parquet includes instance+semantic; split: ann_instance.parquet and ann_semantic.parquet",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # resolve inputs
    if args.input:
        inputs = parse_inputs(args.input)
    else:
        if not args.dataset or not args.root:
            raise ValueError("Provide either repeated --input dataset=/path OR --dataset + --root")
        inputs = [(args.dataset, Path(args.root).expanduser().resolve())]

    # header-only H/W fallback (still fast)
    from dataflow.adapters.utils import read_hw_fast

    # create writers
    writers: Dict[str, _DatasetWriters] = {}

    def _get_writer_key(ds: str) -> str:
        return "__ALL__" if args.out_layout == "merged" else ds

    def _ensure_writers(ds_key: str) -> _DatasetWriters:
        if ds_key in writers:
            return writers[ds_key]

        base = out_dir if ds_key == "__ALL__" else (out_dir / ds_key)
        base.mkdir(parents=True, exist_ok=True)

        meta_out = base / "meta.parquet"

        if args.ann_out_mode == "mixed":
            ann_out = base / "ann.parquet"
            dw = _DatasetWriters(
                meta_writer=ParquetBatchWriter(meta_out, META_SCHEMA, compression=args.compression),
                ann_writer=ParquetBatchWriter(ann_out, ANN_SCHEMA, compression=args.compression),
                ann_inst_writer=None,
                ann_sem_writer=None,
                meta_rows=[],
                ann_rows=[],
                ann_inst_rows=[],
                ann_sem_rows=[],
            )
        else:
            ann_inst_out = base / "ann_instance.parquet"
            ann_sem_out = base / "ann_semantic.parquet"
            dw = _DatasetWriters(
                meta_writer=ParquetBatchWriter(meta_out, META_SCHEMA, compression=args.compression),
                ann_writer=None,
                ann_inst_writer=ParquetBatchWriter(ann_inst_out, ANN_SCHEMA, compression=args.compression),
                ann_sem_writer=ParquetBatchWriter(ann_sem_out, ANN_SCHEMA, compression=args.compression),
                meta_rows=[],
                ann_rows=[],
                ann_inst_rows=[],
                ann_sem_rows=[],
            )

        writers[ds_key] = dw
        return dw

    # process datasets
    for dataset, root in inputs:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        adapter = make_adapter(dataset, root, args)
        ds_key = _get_writer_key(dataset)
        dw = _ensure_writers(ds_key)

        n_checked = 0
        for sample in adapter.iter_samples():
            if args.max_samples > 0 and n_checked >= args.max_samples:
                break
            n_checked += 1
            dw.n_samples += 1

            # ---- H/W: prefer meta, else read header ----
            H, W = _get_hw_from_meta(sample)
            if H is None or W is None:
                h2, w2 = read_hw_fast(str(sample.image_path))
                H, W = int(h2), int(w2)

            slide_uid = f"{dataset}:{sample.sample_id}"

            # ---- mpp: prefer meta, else CLI defaults ----
            mx = _valid_mpp(sample.meta.get("mpp_x")) if sample.meta else None
            my = _valid_mpp(sample.meta.get("mpp_y")) if sample.meta else None
            if mx is None:
                mx = _valid_mpp(args.mpp_x) or _valid_mpp(args.mpp)
            if my is None:
                my = _valid_mpp(args.mpp_y) or _valid_mpp(args.mpp)

            extra_meta = None
            if args.store_extra and sample.meta:
                extra_meta = json.dumps(sample.meta, ensure_ascii=False)

            dw.meta_rows.append(
                {
                    "slide_uid": slide_uid,
                    "dataset": dataset,
                    "sample_id": sample.sample_id,
                    "split": str(sample.split),
                    "group_id": str(sample.group_id),
                    "rel_path": safe_relpath(sample.image_path, root),
                    "backend_type": "PIL",
                    "width_px": int(W),
                    "height_px": int(H),
                    "mpp_x": float(mx) if mx is not None else -1.0,
                    "mpp_y": float(my) if my is not None else -1.0,
                    "tissue_type": str(sample.meta.get("tissue_type")) if sample.meta and sample.meta.get("tissue_type") else None,
                    "stain_type": str(sample.meta.get("stain_type")) if sample.meta and sample.meta.get("stain_type") else None,
                    "source": dataset,
                    "patient_id": str(sample.meta.get("patient_id")) if sample.meta and sample.meta.get("patient_id") else None,
                    "extra": extra_meta,
                }
            )

            # flush meta chunk
            if len(dw.meta_rows) >= args.chunk_size:
                dw.meta_writer.write_pylist(dw.meta_rows)
                dw.meta_rows.clear()

            # ---- semantic union builders (from instance anns) ----
            unions: Dict[Tuple[int, str], _UnionMaskBuilder] = {}
            union_ninst: Dict[Tuple[int, str], int] = {}

            # ---- annotations ----
            for ann in adapter.iter_ann(sample):
                if ann.mask is None:
                    continue

                mask = ann.mask.astype(bool)
                if mask.shape != (H, W):
                    continue

                # bbox for ROI and union
                bbox = ann.bbox_xywh
                if bbox is None:
                    b2 = _bbox_from_mask_np(mask)
                    if b2 is None:
                        continue
                    x0, y0, bw, bh = b2
                else:
                    x0, y0, bw, bh = bbox

                x0, y0, bw, bh = _clip_bbox_xywh(x0, y0, bw, bh, W=W, H=H)
                if bw <= 0 or bh <= 0:
                    continue

                inst_crop = mask[y0 : y0 + bh, x0 : x0 + bw]
                if inst_crop.size == 0 or (not inst_crop.any()):
                    continue

                # update union for derived semantic
                if int(args.emit_semantic_from_instance) == 1 and str(ann.kind) == "instance":
                    if args.semantic_mode == "all":
                        key = (-1, "all")
                    else:
                        lid = int(ann.source_label_id) if ann.source_label_id is not None else -1
                        lname = str(ann.source_label)
                        key = (lid, lname)

                    if key not in unions:
                        unions[key] = _UnionMaskBuilder(x0, y0, inst_crop)
                        union_ninst[key] = 1
                    else:
                        unions[key].update(x0, y0, inst_crop)
                        union_ninst[key] = union_ninst.get(key, 0) + 1

                # choose ROI for RLE storage
                if args.rle_roi == "bbox":
                    roi_x, roi_y, roi_w, roi_h = x0, y0, bw, bh
                    crop = inst_crop
                    rle_h, rle_w = roi_h, roi_w
                else:
                    roi_x, roi_y, roi_w, roi_h = 0, 0, W, H
                    crop = mask
                    rle_h, rle_w = H, W

                counts = mask_to_rle_counts(crop)
                area = int(ann.area) if ann.area is not None else int(np.count_nonzero(crop))

                extra_ann = None
                if args.store_extra:
                    payload: Dict[str, Any] = {}
                    if getattr(ann, "meta", None):
                        payload["ann_meta"] = ann.meta
                    if payload:
                        extra_ann = json.dumps(payload, ensure_ascii=False)

                ann_uid = f"{slide_uid}:{ann.ann_id}"
                row = {
                    "ann_uid": ann_uid,
                    "slide_uid": slide_uid,
                    "dataset": dataset,
                    "task_type": "SEG",
                    "ann_kind": str(ann.kind),
                    "label_name": str(ann.source_label),
                    "label_id": int(ann.source_label_id) if ann.source_label_id is not None else -1,
                    "roi_x": int(roi_x),
                    "roi_y": int(roi_y),
                    "roi_w": int(roi_w),
                    "roi_h": int(roi_h),
                    "geometry_fmt": "RLE",
                    "rle_order": "F",
                    "rle_size_h": int(rle_h),
                    "rle_size_w": int(rle_w),
                    "rle_counts": counts,
                    "bbox_x": int(x0),
                    "bbox_y": int(y0),
                    "bbox_w": int(bw),
                    "bbox_h": int(bh),
                    "area": int(area),
                    "is_gt": True,
                    "confidence": float(getattr(ann, "confidence", 1.0)) if getattr(ann, "confidence", None) is not None else 1.0,
                    "annotator": dataset,
                    "extra": extra_ann,
                }
                dw.emit_ann_row(row, args)

            # ---- emit derived semantic rows at end of sample ----
            if int(args.emit_semantic_from_instance) == 1 and unions:
                for (lid, lname), ub in unions.items():
                    sem_mask_roi = ub.mask
                    sem_area = int(np.count_nonzero(sem_mask_roi))
                    if sem_area < int(args.semantic_min_area):
                        continue

                    ux, uy, uw, uh = ub.roi_x, ub.roi_y, ub.roi_w, ub.roi_h

                    if args.rle_roi == "bbox":
                        roi_x, roi_y, roi_w, roi_h = ux, uy, uw, uh
                        crop = sem_mask_roi
                        rle_h, rle_w = uh, uw
                    else:
                        roi_x, roi_y, roi_w, roi_h = 0, 0, W, H
                        full = np.zeros((H, W), dtype=bool)
                        full[uy : uy + uh, ux : ux + uw] = sem_mask_roi
                        crop = full
                        rle_h, rle_w = H, W

                    counts = mask_to_rle_counts(crop)
                    safe = _safe_label(lname)
                    ann_uid = f"{slide_uid}:SEM:{lid}:{safe}"

                    extra_sem = None
                    if args.store_extra:
                        extra_sem = json.dumps(
                            {
                                "derived_from": "instance_union",
                                "semantic_mode": args.semantic_mode,
                                "n_instances": int(union_ninst.get((lid, lname), 0)),
                            },
                            ensure_ascii=False,
                        )

                    sem_row = {
                        "ann_uid": ann_uid,
                        "slide_uid": slide_uid,
                        "dataset": dataset,
                        "task_type": "SEG",
                        "ann_kind": "semantic",
                        "label_name": str(lname),
                        "label_id": int(lid),
                        "roi_x": int(roi_x),
                        "roi_y": int(roi_y),
                        "roi_w": int(roi_w),
                        "roi_h": int(roi_h),
                        "geometry_fmt": "RLE",
                        "rle_order": "F",
                        "rle_size_h": int(rle_h),
                        "rle_size_w": int(rle_w),
                        "rle_counts": counts,
                        "bbox_x": int(ux),
                        "bbox_y": int(uy),
                        "bbox_w": int(uw),
                        "bbox_h": int(uh),
                        "area": int(sem_area),
                        "is_gt": True,
                        "confidence": 1.0,
                        "annotator": dataset,
                        "extra": extra_sem,
                    }
                    dw.emit_ann_row(sem_row, args)

        print(f"[OK] {dataset}: processed_samples={n_checked}")

    # close all writers + flush
    print("\n========== DONE ==========")
    for k, dw in writers.items():
        dw.close(args)
        base = out_dir if k == "__ALL__" else (out_dir / k)
        print(f"- {k}: out={base} samples={dw.n_samples} anns={dw.n_anns}")

    print(f"\nroot_out_dir: {out_dir}")
    print("If out_layout=per_dataset, check subfolders; if merged, check files directly under out_dir.")


if __name__ == "__main__":
    main()
