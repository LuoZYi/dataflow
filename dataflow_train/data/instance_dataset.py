# dataflow_train/data/instance_dataset.py
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from dataflow_train.utils.rle import rle_decode_counts  # 你已有的话；没有就把你 semantic 里那段搬到 utils/rle.py


def _safe_relpath_join(root: Path, rel_path: str) -> Path:
    return (root / rel_path).resolve()


def _norm_split(s: str) -> str:
    s = str(s).lower()
    if s in ("val", "valid", "validation"):
        return "val"
    if s in ("train", "tr"):
        return "train"
    if s in ("test", "te"):
        return "test"
    return s


def _compute_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """
    inst_map: [H,W] int32 (0=bg, 1..K instances)
    Returns hv: [2,H,W] float32 in [-1,1], masked outside nuclei (0)
    """
    H, W = inst_map.shape
    hv = np.zeros((2, H, W), dtype=np.float32)
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    for iid in ids:
        ys, xs = np.where(inst_map == iid)
        if len(xs) < 5:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        # normalize by max abs to keep [-1,1]
        nx = float(np.max(np.abs(dx)) + 1e-6)
        ny = float(np.max(np.abs(dy)) + 1e-6)
        dxn = dx / nx
        dyn = dy / ny
        hv[0, ys, xs] = dxn
        hv[1, ys, xs] = dyn
    return hv


class ParquetInstanceCropDataset(Dataset):
    """
    Returns:
      image: [3,ps,ps] float
      inst_map: [ps,ps] long (0=bg, 1..K local ids)
      nuclei: [1,ps,ps] float {0,1}
      hv: [2,ps,ps] float
      slide_uid, crop_xy, dataset
    """

    def __init__(
        self,
        db_root: str,
        dataset_roots: Dict[str, str],
        datasets: List[str],
        split: str,
        ann_file: str = "ann_instance.parquet",
        patch_size: int = 512,
        epoch_size: int = 4000,
        max_slides: int = 0,
        pos_fraction: float = 0.8,
        seed: int = 42,
        cache_slides: int = 64,
        use_meta_split: bool = True,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.db_root = Path(db_root)
        self.dataset_roots = {k: Path(v) for k, v in dataset_roots.items()}
        self.datasets = [d.lower() for d in datasets]
        self.split = _norm_split(split)
        self.ann_file = ann_file
        self.patch_size = int(patch_size)
        self.epoch_size = int(epoch_size)
        self.pos_fraction = float(pos_fraction)
        self.rng = random.Random(int(seed))
        self.cache_slides = int(cache_slides)
        self.use_meta_split = bool(use_meta_split)
        self.val_ratio = float(val_ratio)

        meta_list = []
        ann_list = []

        for ds in self.datasets:
            meta_path = self.db_root / ds / "meta.parquet"
            ann_path = self.db_root / ds / ann_file
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing: {meta_path}")
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing: {ann_path}")

            meta = pd.read_parquet(
                meta_path,
                columns=["slide_uid", "dataset", "rel_path", "width_px", "height_px", "split"],
            )
            ann = pd.read_parquet(
                ann_path,
                columns=[
                    "ann_uid", "slide_uid",
                    "label_id", "label_name",
                    "roi_x", "roi_y", "roi_w", "roi_h",
                    "rle_size_h", "rle_size_w", "rle_counts", "area"
                ],
            )
            meta_list.append(meta)
            ann_list.append(ann)

        meta_all = pd.concat(meta_list, ignore_index=True)
        ann_all = pd.concat(ann_list, ignore_index=True)

        meta_all["split"] = meta_all["split"].map(_norm_split)

        if not self.use_meta_split:
            # deterministic hash split by slide_uid
            uids = meta_all["slide_uid"].astype(str).tolist()
            keep = []
            for i, suid in enumerate(uids):
                h = (hash(suid) % 100000) / 100000.0
                sp = "val" if h < self.val_ratio else "train"
                keep.append(sp)
            meta_all["split"] = keep

        meta_all = meta_all[meta_all["split"] == self.split].reset_index(drop=True)
        if max_slides and max_slides > 0:
            meta_all = meta_all.head(int(max_slides)).reset_index(drop=True)

        self.meta = meta_all
        self.slide_uids = self.meta["slide_uid"].astype(str).tolist()

        # group anns by slide
        self.ann_by_slide: Dict[str, List[Dict[str, Any]]] = {}
        for r in ann_all.to_dict("records"):
            suid = str(r["slide_uid"])
            self.ann_by_slide.setdefault(suid, []).append(r)

        # cache decoded rois per slide: [(rx,ry,mask_bool,label_id)]
        self._slide_cache: "OrderedDict[str, List[Tuple[int,int,np.ndarray,int]]]" = OrderedDict()

    def __len__(self):
        return self.epoch_size if self.split == "train" else len(self.slide_uids)

    def _load_image_full(self, mr: Dict[str, Any]) -> np.ndarray:
        ds = str(mr["dataset"]).lower()
        root = self.dataset_roots.get(ds, None)
        if root is None:
            raise KeyError(f"dataset_roots missing key: {ds}")
        p = _safe_relpath_join(root, str(mr["rel_path"]))
        img = Image.open(p).convert("RGB")
        return np.array(img)

    def _get_slide_decoded_instances(self, slide_uid: str) -> List[Tuple[int, int, np.ndarray, int]]:
        if slide_uid in self._slide_cache:
            self._slide_cache.move_to_end(slide_uid)
            return self._slide_cache[slide_uid]

        anns = self.ann_by_slide.get(slide_uid, [])
        decoded: List[Tuple[int, int, np.ndarray, int]] = []
        for a in anns:
            counts = a.get("rle_counts", None)
            if counts is None:
                continue
            rx, ry = int(a["roi_x"]), int(a["roi_y"])
            h, w = int(a["rle_size_h"]), int(a["rle_size_w"])
            m = rle_decode_counts(counts, h, w)  # bool [h,w]
            label_id = int(a.get("label_id", 1) or 1)
            decoded.append((rx, ry, m, label_id))

        self._slide_cache[slide_uid] = decoded
        self._slide_cache.move_to_end(slide_uid)
        while len(self._slide_cache) > self.cache_slides:
            self._slide_cache.popitem(last=False)
        return decoded

    def _sample_crop_xy(self, slide_uid: str, H: int, W: int) -> Tuple[int, int]:
        ps = self.patch_size
        if H <= ps or W <= ps:
            return 0, 0

        if self.split == "train" and self.rng.random() < self.pos_fraction:
            insts = self._get_slide_decoded_instances(slide_uid)
            if insts:
                rx, ry, m, _ = self.rng.choice(insts)
                mh, mw = m.shape
                xmin = max(0, rx - ps + 1)
                xmax = min(W - ps, rx + mw - 1)
                ymin = max(0, ry - ps + 1)
                ymax = min(H - ps, ry + mh - 1)
                if xmin <= xmax and ymin <= ymax:
                    return self.rng.randint(xmin, xmax), self.rng.randint(ymin, ymax)

        return self.rng.randint(0, W - ps), self.rng.randint(0, H - ps)

    def _make_crop_inst_map(self, slide_uid: str, cx: int, cy: int, ps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build local inst_map and nuclei mask in crop.
        inst_map: [ps,ps] int32 (0 bg, 1..K)
        nuclei: [ps,ps] bool
        """
        inst_map = np.zeros((ps, ps), dtype=np.int32)
        insts = self._get_slide_decoded_instances(slide_uid)
        x0, y0 = int(cx), int(cy)
        x1, y1 = x0 + ps, y0 + ps

        cur_id = 0
        for (rx, ry, roi_mask, _label_id) in insts:
            mh, mw = roi_mask.shape
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
            if sub.size == 0 or sub.sum() == 0:
                continue

            cx0 = ix0 - x0
            cy0 = iy0 - y0

            # allocate a new local id
            cur_id += 1
            # only write where empty (avoid overlap)
            target = inst_map[cy0:cy0 + sub.shape[0], cx0:cx0 + sub.shape[1]]
            target[(sub > 0) & (target == 0)] = cur_id

        nuclei = inst_map > 0
        return inst_map, nuclei

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == "train":
            slide_uid = self.slide_uids[idx % len(self.slide_uids)]
        else:
            slide_uid = self.slide_uids[idx]

        mr = self.meta[self.meta["slide_uid"].astype(str) == slide_uid].iloc[0].to_dict()
        img_full = self._load_image_full(mr)
        H, W = img_full.shape[:2]
        ps = self.patch_size

        cx, cy = self._sample_crop_xy(slide_uid, H, W)
        crop = img_full[cy:cy + ps, cx:cx + ps]
        if crop.shape[0] != ps or crop.shape[1] != ps:
            pad = np.zeros((ps, ps, 3), dtype=np.uint8)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad

        inst_map, nuclei = self._make_crop_inst_map(slide_uid, cx, cy, ps)
        hv = _compute_hv_map(inst_map)

        img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0          # [3,ps,ps]
        inst_t = torch.from_numpy(inst_map.astype(np.int64))                     # [ps,ps]
        nuc_t = torch.from_numpy(nuclei.astype(np.uint8))[None].float()          # [1,ps,ps]
        hv_t = torch.from_numpy(hv)                                              # [2,ps,ps]

        return {
            "image": img_t,
            "inst_map": inst_t,
            "nuclei": nuc_t,
            "hv": hv_t,
            "slide_uid": str(slide_uid),
            "crop_xy": (int(cx), int(cy)),
            "full_hw": (int(H), int(W)),
            "dataset": str(mr["dataset"]),
        }