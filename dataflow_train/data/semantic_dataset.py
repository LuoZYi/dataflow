# dataflow_train/data/semantic_dataset.py
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from dataflow_train.utils.rle import rle_decode_counts


def _norm_split(s: Any) -> str:
    s = str(s).lower()
    if s in ("val", "valid", "validation"):
        return "val"
    if s in ("train", "tr"):
        return "train"
    if s in ("test", "te"):
        return "test"
    return s


def _safe_relpath_join(root: Path, rel_path: str) -> Path:
    return (root / rel_path).resolve()


def _stable_u01(uid: str) -> float:
    # stable-ish hash to [0,1)
    h = 2166136261
    for ch in uid:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % 1000000) / 1000000.0


@dataclass
class SemanticSample:
    image: torch.Tensor   # [3,ps,ps] float32
    mask: torch.Tensor    # [1,ps,ps] float32 {0,1}
    slide_uid: str
    crop_xy: Tuple[int, int]
    full_hw: Tuple[int, int]
    dataset: str


class ParquetSemanticCropDataset(Dataset):
    """
    Parquet semantic dataset with rle_roi=bbox.

    Key invariant:
      crop_mask == crop(full_mask(paste all ROI masks))

    We build crop mask directly by intersecting each ROI with crop window.
    """

    def __init__(
        self,
        db_root: str,
        dataset_roots: Dict[str, str],
        datasets: List[str],
        split: str,
        ann_file: str = "ann_semantic.parquet",
        patch_size: int = 512,
        epoch_size: int = 2000,
        max_slides: int = 0,
        pos_fraction: float = 0.7,
        seed: int = 42,
        cache_slides: int = 64,
        use_meta_split: bool = True,
        val_ratio: float = 0.1,
        image_mode: str = "RGB",
    ):
        super().__init__()
        self.db_root = Path(db_root)
        self.dataset_roots = {k.lower(): Path(v) for k, v in dataset_roots.items()}
        self.datasets = [d.lower() for d in datasets]
        self.split = _norm_split(split)
        self.ann_file = ann_file
        self.patch_size = int(patch_size)
        self.epoch_size = int(epoch_size)
        self.pos_fraction = float(pos_fraction)
        self.rng = random.Random(seed)
        self.cache_slides = int(cache_slides)
        self.use_meta_split = bool(use_meta_split)
        self.val_ratio = float(val_ratio)
        self.image_mode = image_mode

        meta_list = []
        ann_list = []

        for ds in self.datasets:
            meta_path = self.db_root / ds / "meta.parquet"
            ann_path = self.db_root / ds / ann_file
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing: {meta_path}")
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing: {ann_path}")

            meta = pd.read_parquet(meta_path)

            # required
            for need in ["slide_uid", "rel_path"]:
                if need not in meta.columns:
                    raise KeyError(f"{meta_path} missing {need}")

            if "dataset" not in meta.columns:
                meta["dataset"] = ds

            if "height_px" not in meta.columns and "H" in meta.columns:
                meta["height_px"] = meta["H"]
            if "width_px" not in meta.columns and "W" in meta.columns:
                meta["width_px"] = meta["W"]
            if "height_px" not in meta.columns or "width_px" not in meta.columns:
                raise KeyError(f"{meta_path} missing height_px/width_px")

            if "split" in meta.columns:
                meta["split"] = meta["split"].map(_norm_split)
            else:
                meta["split"] = "unknown"

            meta_list.append(meta[["slide_uid", "dataset", "rel_path", "width_px", "height_px", "split"]].copy())

            ann = pd.read_parquet(ann_path)
            for need in ["slide_uid", "roi_x", "roi_y", "rle_size_h", "rle_size_w", "rle_counts"]:
                if need not in ann.columns:
                    raise KeyError(f"{ann_path} missing {need}")

            ann_list.append(ann[["slide_uid", "roi_x", "roi_y", "rle_size_h", "rle_size_w", "rle_counts"]].copy())

        meta_all = pd.concat(meta_list, ignore_index=True)
        ann_all = pd.concat(ann_list, ignore_index=True)

        # split selection
        if self.use_meta_split and (meta_all["split"] != "unknown").any():
            meta_all["split"] = meta_all["split"].map(_norm_split)
            meta_sel = meta_all[meta_all["split"] == self.split].reset_index(drop=True)
        else:
            # deterministic split by uid hash
            meta_all = meta_all.copy()
            tags = []
            for uid in meta_all["slide_uid"].astype(str).tolist():
                u = _stable_u01(uid)
                tags.append("val" if u < self.val_ratio else "train")
            meta_all["split"] = tags
            meta_sel = meta_all[meta_all["split"] == self.split].reset_index(drop=True)

        if max_slides and max_slides > 0:
            meta_sel = meta_sel.head(int(max_slides)).reset_index(drop=True)

        self.meta = meta_sel
        self.slide_uids: List[str] = self.meta["slide_uid"].astype(str).tolist()

        # group annotations per slide
        self.ann_by_slide: Dict[str, List[Dict[str, Any]]] = {}
        for r in ann_all.to_dict("records"):
            suid = str(r["slide_uid"])
            self.ann_by_slide.setdefault(suid, []).append(r)

        # LRU cache: slide_uid -> decoded ROI masks [(rx,ry,mask_bool,w,h)]
        self._slide_cache: Dict[str, Any] = {}
        self._lru: List[str] = []

    def __len__(self):
        return self.epoch_size if self.split == "train" else len(self.slide_uids)

    def _load_image_full(self, meta_row: Dict[str, Any]) -> np.ndarray:
        ds = str(meta_row["dataset"]).lower()
        root = self.dataset_roots.get(ds)
        if root is None:
            raise KeyError(f"dataset_roots missing key: {ds}")

        p = _safe_relpath_join(root, str(meta_row["rel_path"]))
        try:
            with Image.open(p) as im:
                im = im.convert(self.image_mode)
                return np.array(im)
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {p}") from e

    def _cache_put(self, slide_uid: str, decoded):
        if slide_uid in self._slide_cache:
            return
        self._slide_cache[slide_uid] = decoded
        self._lru.append(slide_uid)
        if len(self._lru) > self.cache_slides:
            old = self._lru.pop(0)
            self._slide_cache.pop(old, None)

    def _get_slide_decoded_rois(self, slide_uid: str):
        if slide_uid in self._slide_cache:
            return self._slide_cache[slide_uid]

        anns = self.ann_by_slide.get(slide_uid, [])
        decoded = []
        for a in anns:
            rx, ry = int(a["roi_x"]), int(a["roi_y"])
            h, w = int(a["rle_size_h"]), int(a["rle_size_w"])
            counts = a["rle_counts"]
            if counts is None:
                continue
            m = rle_decode_counts(counts, h, w)
            decoded.append((rx, ry, m, w, h))

        self._cache_put(slide_uid, decoded)
        return decoded

    def _sample_crop_xy(self, slide_uid: str, H: int, W: int) -> Tuple[int, int]:
        ps = self.patch_size
        if H <= ps or W <= ps:
            return 0, 0

        if self.split == "train" and (self.rng.random() < self.pos_fraction):
            rois = self._get_slide_decoded_rois(slide_uid)
            if rois:
                rx, ry, m, mw, mh = self.rng.choice(rois)
                xmin = max(0, rx - ps + 1)
                xmax = min(W - ps, rx + mw - 1)
                ymin = max(0, ry - ps + 1)
                ymax = min(H - ps, ry + mh - 1)
                if xmin <= xmax and ymin <= ymax:
                    return self.rng.randint(xmin, xmax), self.rng.randint(ymin, ymax)

        return self.rng.randint(0, W - ps), self.rng.randint(0, H - ps)

    def _make_crop_mask(self, slide_uid: str, cx: int, cy: int, ps: int) -> np.ndarray:
        crop_mask = np.zeros((ps, ps), dtype=bool)
        x0, y0 = cx, cy
        x1, y1 = x0 + ps, y0 + ps

        for (rx, ry, roi_mask, mw, mh) in self._get_slide_decoded_rois(slide_uid):
            ax0, ay0 = rx, ry
            ax1, ay1 = rx + mw, ry + mh

            ix0 = max(x0, ax0)
            iy0 = max(y0, ay0)
            ix1 = min(x1, ax1)
            iy1 = min(y1, ay1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            roi_sx0 = ix0 - ax0
            roi_sy0 = iy0 - ay0
            roi_sx1 = ix1 - ax0
            roi_sy1 = iy1 - ay0
            sub = roi_mask[roi_sy0:roi_sy1, roi_sx0:roi_sx1]
            if sub.size == 0:
                continue

            cx0 = ix0 - x0
            cy0 = iy0 - y0
            cx1 = cx0 + sub.shape[1]
            cy1 = cy0 + sub.shape[0]
            crop_mask[cy0:cy1, cx0:cx1] |= sub

        return crop_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == "train":
            slide_uid = self.slide_uids[idx % len(self.slide_uids)]
        else:
            slide_uid = self.slide_uids[idx]

        mr = self.meta[self.meta["slide_uid"].astype(str) == slide_uid].iloc[0].to_dict()
        img_full = self._load_image_full(mr)
        H, W = img_full.shape[0], img_full.shape[1]
        ps = self.patch_size

        if self.split == "train":
            cx, cy = self._sample_crop_xy(slide_uid, H, W)
        else:
            # 稳定 val：center crop
            cx = max(0, (W - ps) // 2) if W > ps else 0
            cy = max(0, (H - ps) // 2) if H > ps else 0

        crop = img_full[cy:cy + ps, cx:cx + ps]
        if crop.shape[0] != ps or crop.shape[1] != ps:
            pad = np.zeros((ps, ps, 3), dtype=np.uint8)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad

        crop_mask = self._make_crop_mask(slide_uid, cx, cy, ps).astype(np.uint8)

        img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        m_t = torch.from_numpy(crop_mask)[None, ...].float()

        return {
            "image": img_t,
            "mask": m_t,
            "slide_uid": slide_uid,
            "crop_xy": (int(cx), int(cy)),
            "full_hw": (int(H), int(W)),
            "dataset": str(mr["dataset"]),
        }
