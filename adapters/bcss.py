# dataflow/adapters/bcss.py
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .types import AnnObject, BaseAdapter, Sample
from .utils import index_by_stem, read_rgb, read_mask_any, bbox_from_mask, area_from_mask, read_hw_fast


def _read_gtruth_codes_tsv(tsv_path: Path) -> Dict[int, str]:
    """
    BCSS official repo provides meta/gtruth_codes.tsv mapping id->name.
    We'll parse if present; otherwise fallback.
    """
    out: Dict[int, str] = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        for row in r:
            if not row or len(row) < 2:
                continue
            try:
                k = int(row[0])
            except Exception:
                continue
            out[k] = row[1].strip()
    return out


class BCSSAdapter(BaseAdapter):
    dataset_name = "bcss"

    def __init__(
        self,
        root: Path,
        *,
        images_dir: str = "rgbs_colorNormalized",
        masks_dir: str = "masks",
        split_mode: str = "unspecified",  # "unspecified" | "random"
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        gtruth_tsv: Optional[str] = "meta/gtruth_codes.tsv",
    ) -> None:
        super().__init__(root)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.split_mode = split_mode
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gtruth_tsv = gtruth_tsv

        self.id2name: Dict[int, str] = {
            1: "tumor",
            2: "stroma",
            3: "lym",
            4: "necrosis",
            5: "other",
        }
        if gtruth_tsv:
            p = self.root / gtruth_tsv
            if p.exists():
                self.id2name.update(_read_gtruth_codes_tsv(p))

    def _assign_splits(self, sample_ids: List[str]) -> Dict[str, str]:
        if self.split_mode != "random":
            return {sid: "unspecified" for sid in sample_ids}

        rng = random.Random(self.seed)
        ids = list(sample_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * self.test_ratio))
        n_val = int(round(n * self.val_ratio))
        test = set(ids[:n_test])
        val = set(ids[n_test:n_test + n_val])

        out: Dict[str, str] = {}
        for sid in ids:
            if sid in test:
                out[sid] = "test"
            elif sid in val:
                out[sid] = "val"
            else:
                out[sid] = "train"
        return out

    def iter_samples(self) -> Iterator[Sample]:
        img_dir = self.root / self.images_dir
        msk_dir = self.root / self.masks_dir

        img_map = index_by_stem(img_dir)
        msk_map = index_by_stem(msk_dir)

        common = sorted(set(img_map.keys()) & set(msk_map.keys()))
        if not common:
            raise ValueError(f"[BCSS] No matched images/masks under {img_dir} and {msk_dir}")

        split_of = self._assign_splits(common)

        for stem in common:
            sp = split_of[stem]
            H, W = read_hw_fast(str(img_map[stem]))
            yield Sample(
                dataset=self.dataset_name,
                sample_id=stem,
                split=sp,  # type: ignore
                image_path=img_map[stem],
                ann_path=msk_map[stem],
                group_id=stem,
                meta={
                    "label_map": dict(self.id2name),
                    "height_px": H,
                    "width_px": W,
                    "mpp_x": 0.250,
                    "mpp_y": 0.250,
                },
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        assert sample.ann_path is not None
        mask_arr = read_mask_any(sample.ann_path)

        if mask_arr.ndim == 3:
            # some dumps might be RGB; convert to grayscale by first channel if identical
            if np.all(mask_arr[..., 0] == mask_arr[..., 1]) and np.all(mask_arr[..., 0] == mask_arr[..., 2]):
                mask_arr = mask_arr[..., 0]
            else:
                # fallback: treat each unique color (rare for BCSS)
                packed = (mask_arr[..., 0].astype(np.int32) << 16) + (mask_arr[..., 1].astype(np.int32) << 8) + mask_arr[..., 2].astype(np.int32)
                uniq = [u for u in np.unique(packed).tolist() if u != 0]
                for u in uniq:
                    m = packed == u
                    yield AnnObject(
                        ann_id=f"{sample.sample_id}:rgb{u}",
                        kind="semantic",
                        source_label=f"rgb_{u}",
                        mask=m,
                        bbox_xywh=bbox_from_mask(m),
                        area=area_from_mask(m),
                    )
                return

        # Normal BCSS: pixel values are class ids; 0 = outside ROI (don't-care)
        uniq_ids = [int(x) for x in np.unique(mask_arr).tolist() if int(x) != 0]
        for cid in uniq_ids:
            m = (mask_arr == cid)
            name = sample.meta.get("label_map", {}).get(cid, f"class_{cid}")
            yield AnnObject(
                ann_id=f"{sample.sample_id}:c{cid}",
                kind="semantic",
                source_label=str(name),
                source_label_id=cid,
                mask=m,
                bbox_xywh=bbox_from_mask(m),
                area=area_from_mask(m),
            )
