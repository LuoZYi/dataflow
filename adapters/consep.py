# dataflow/adapters/consep.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import scipy.io as sio

from .types import AnnObject, BaseAdapter, Sample
from .utils import index_by_stem, read_rgb, find_instance_slices, bbox_from_mask, area_from_mask, read_hw_fast


_CONSEP_TYPE_ID2NAME = {
    1: "misc",          # miscellaneous
    2: "inflammatory",
    3: "epithelial",
    4: "spindle",
}


def _pick_key(d: Dict[str, Any], candidates: list[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


class ConSepAdapter(BaseAdapter):
    dataset_name = "consep"

    def __init__(self, root: Path, *, type_id2name: Optional[Dict[int, str]] = None) -> None:
        super().__init__(root)
        self.type_id2name = type_id2name or _CONSEP_TYPE_ID2NAME

    def iter_samples(self) -> Iterator[Sample]:
        root = self.root
        for split_dir, split_name in [("Train", "train"), ("Test", "test")]:
            img_dir = root / split_dir / "Images"
            ann_dir = root / split_dir / "Labels"
            if not img_dir.exists():
                continue

            img_map = index_by_stem(img_dir)
            ann_map = {p.stem: p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mat"}

            common = sorted(set(img_map.keys()) & set(ann_map.keys()))
            if not common:
                raise ValueError(f"[ConSep] No matched image/mat under {img_dir} and {ann_dir}")

            for stem in common:
                ip = img_map[stem]
                H, W = read_hw_fast(str(ip))  # fast header read, cached
                yield Sample(
                    dataset=self.dataset_name,
                    sample_id=f"{split_name}:{stem}",
                    split=split_name,  # type: ignore
                    image_path=ip,
                    ann_path=ann_map[stem],
                    group_id=stem,
                    meta={
                        "split_dir": split_dir,
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
        mat = sio.loadmat(sample.ann_path)

        inst_key = _pick_key(mat, ["inst_map", "inst", "instance_map", "instMap"])
        if inst_key is None:
            raise ValueError(f"[ConSep] inst_map not found in {sample.ann_path}. keys={list(mat.keys())[:20]}")

        inst_map = mat[inst_key].astype(np.int32)
        type_map = None
        tkey = _pick_key(mat, ["type_map", "type", "class_map", "typeMap"])
        if tkey is not None:
            type_map = mat[tkey].astype(np.int32)

        slices = find_instance_slices(inst_map)
        # label id is (idx+1)
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            sub = inst_map[slc]
            m = (sub == idx)
            if not np.any(m):
                continue

            # decide type: prefer per-pixel type_map majority if available
            source_label_id: Optional[int] = None
            if type_map is not None:
                sub_t = type_map[slc]
                vals, cnts = np.unique(sub_t[m], return_counts=True)
                # remove background 0 if it exists
                keep = vals != 0
                if np.any(keep):
                    vals, cnts = vals[keep], cnts[keep]
                if vals.size > 0:
                    source_label_id = int(vals[np.argmax(cnts)])

            label_name = (
                self.type_id2name.get(source_label_id, f"class_{source_label_id}")
                if source_label_id is not None
                else "unknown"
            )

            # bbox in full image coords
            y0, y1 = slc[0].start, slc[0].stop
            x0, x1 = slc[1].start, slc[1].stop
            full_mask = np.zeros(inst_map.shape, dtype=bool)
            full_mask[y0:y1, x0:x1] = m

            yield AnnObject(
                ann_id=f"{sample.sample_id}:{idx}",
                kind="instance",
                source_label=label_name,
                source_label_id=source_label_id,
                mask=full_mask,
                bbox_xywh=bbox_from_mask(full_mask),
                area=area_from_mask(full_mask),
            )
