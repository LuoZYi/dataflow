
# dataflow/adapters/pannuke.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .types import AnnObject, BaseAdapter, Sample
from .utils import read_rgb, read_hw_fast, find_instance_slices, bbox_from_mask, area_from_mask



# Default mapping for PanNuke nucleus types (pixel-level type_id in type_map)
_PANNUKE_ID2NAME: Dict[int, str] = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def _infer_inst_and_type(arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Infer (inst_map, type_map) from PanNuke label .npy variants.

    Supported:
      1) Pickled dict stored as 0-d object ndarray:
         {"inst_map": (H,W), "type_map": (H,W)}
      2) (H,W) int map: treated as inst_map only
      3) (H,W,2): [:,:,0]=inst_map, [:,:,1]=type_map
      4) (H,W,C): heuristic (channel with largest max is inst_map; other channels infer type_map)
    """
    # ---- Case 1: 0-d object array containing a dict ----
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        obj = arr.item()
        if not isinstance(obj, dict):
            raise ValueError(f"Unsupported PanNuke label object type: {type(obj)}")

        if "inst_map" not in obj:
            raise ValueError(f"PanNuke dict missing 'inst_map'. keys={list(obj.keys())}")

        inst_map = np.asarray(obj["inst_map"]).astype(np.int32)
        type_map = None
        if "type_map" in obj and obj["type_map"] is not None:
            type_map = np.asarray(obj["type_map"]).astype(np.int32)

        if inst_map.ndim != 2:
            raise ValueError(f"PanNuke inst_map must be 2D, got shape={inst_map.shape}")
        if type_map is not None and type_map.shape != inst_map.shape:
            # keep robust: ignore invalid type_map rather than crashing
            type_map = None

        return inst_map, type_map

    # ---- Case 2: (H,W) ----
    if arr.ndim == 2:
        return arr.astype(np.int32), None

    # ---- Case 3: (H,W,2) ----
    if arr.ndim == 3 and arr.shape[2] == 2:
        inst = arr[:, :, 0].astype(np.int32)
        typ = arr[:, :, 1].astype(np.int32)
        return inst, typ

    # ---- Case 4: (H,W,C) heuristic ----
    if arr.ndim == 3:
        # choose channel with largest max as inst_map
        mx = [int(arr[:, :, c].max()) for c in range(arr.shape[2])]
        inst_c = int(np.argmax(mx))
        inst = arr[:, :, inst_c].astype(np.int32)

        rest = [c for c in range(arr.shape[2]) if c != inst_c]
        if not rest:
            return inst, None

        rest_stack = arr[:, :, rest]
        uniq = np.unique(rest_stack)
        # one-hot-ish => build per-pixel type_id by argmax
        if uniq.size <= 3 and set(uniq.tolist()).issubset({0, 1}):
            type_id = (np.argmax(rest_stack, axis=2) + 1).astype(np.int32)
            all0 = np.all(rest_stack == 0, axis=2)
            type_id[all0] = 0
            return inst, type_id

        # otherwise maybe one remaining channel is already type_id (small max)
        small = [(c, int(arr[:, :, c].max())) for c in rest]
        small = [x for x in small if x[1] <= 10]
        if small:
            c = small[0][0]
            return inst, arr[:, :, c].astype(np.int32)

        return inst, None

    raise ValueError(f"Unsupported label array shape: {getattr(arr, 'shape', None)}")


def _majority_vote_type(type_map: np.ndarray, inst_mask: np.ndarray) -> Optional[int]:
    """
    Majority vote the type_id within an instance mask.
    type_id 0 is treated as background and ignored.
    """
    vals = type_map[inst_mask]
    vals = vals[vals != 0]
    if vals.size == 0:
        return None

    # type ids are small (<=5), bincount is efficient
    bc = np.bincount(vals.astype(np.int64))
    if bc.size == 0:
        return None
    return int(bc.argmax())


class PanNukeAdapter(BaseAdapter):
    dataset_name = "pannuke"

    def __init__(
        self,
        root: Path,
        *,
        fold_dirs: Tuple[str, ...] = ("fold0", "fold1", "fold2"),
        fold_to_split: Optional[Dict[str, str]] = None,
        type_id2name: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__(root)
        self.fold_dirs = fold_dirs
        self.fold_to_split = fold_to_split or {"fold0": "train", "fold1": "val", "fold2": "test"}
        self.type_id2name = type_id2name or _PANNUKE_ID2NAME

        # optional meta from types.csv if present (img filename includes .png)
        self._types_csv_cache: Dict[str, str] = {}
        tcsv = self.root / "types.csv"
        if tcsv.exists():
            with tcsv.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    # columns: img,type
                    self._types_csv_cache[row["img"]] = row["type"]

    def iter_samples(self) -> Iterator[Sample]:
        root = self.root
        for fd in self.fold_dirs:
            fold_root = root / fd
            img_dir = fold_root / "images"
            lbl_dir = fold_root / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
            for ip in imgs:
                img_name = ip.name  # keep "0_1.png"
                ap = lbl_dir / (ip.stem + ".npy")
                if not ap.exists():
                    continue

                sp = self.fold_to_split.get(fd, "unspecified")
                H, W = read_hw_fast(str(ip))  # fast header read, cached
                yield Sample(
                    dataset=self.dataset_name,
                    sample_id=f"{fd}:{ip.stem}",
                    split=sp,  # type: ignore
                    image_path=ip,
                    ann_path=ap,
                    group_id=ip.stem,
                    meta={
                        "fold": fd,
                        "tissue_type": self._types_csv_cache.get(img_name),
                        "height_px": H,
                        "width_px": W,
                        "mpp_x": 0.25,
                        "mpp_y": 0.25,
                    },
                )

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        assert sample.ann_path is not None

        arr = np.load(sample.ann_path, allow_pickle=True)
        inst_map, type_map = _infer_inst_and_type(arr)

        slices = find_instance_slices(inst_map)
        H, W = inst_map.shape

        for inst_id, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            sub = inst_map[slc]
            m = (sub == inst_id)
            if not np.any(m):
                continue

            cls_id: Optional[int] = None
            if type_map is not None:
                sub_t = type_map[slc]
                cls_id = _majority_vote_type(sub_t, m)

            label = self.type_id2name.get(cls_id, f"class_{cls_id}") if cls_id is not None else "unknown"

            # place sub-mask back to full image coords
            y0, y1 = slc[0].start, slc[0].stop
            x0, x1 = slc[1].start, slc[1].stop
            full = np.zeros((H, W), dtype=bool)
            full[y0:y1, x0:x1] = m

            yield AnnObject(
                ann_id=f"{sample.sample_id}:n{inst_id}",
                kind="instance",
                source_label=label,
                source_label_id=cls_id,
                mask=full,
                bbox_xywh=bbox_from_mask(full),
                area=area_from_mask(full),
                meta={
                    "tissue_type": sample.meta.get("tissue_type"),
                    "fold": sample.meta.get("fold"),
                    
                },
            )
