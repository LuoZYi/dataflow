# dataflow/adapters/lizard.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import scipy.io as sio

from .types import AnnObject, BaseAdapter, Sample
from .utils import read_rgb, find_instance_slices, bbox_from_mask, area_from_mask, read_hw_fast


class LizardAdapter(BaseAdapter):
    dataset_name = "lizard"

    def __init__(
        self,
        root: Path,
        *,
        images_dir: str = "images",
        labels_dir: str = "mask/Labels",
        info_csv: str = "mask/info.csv",
        split_map: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(root)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.info_csv = info_csv
        # Default guess: 1/2/3 -> train/val/test (你也可以在外面覆盖)
        self.split_map = split_map or {"1": "train", "2": "val", "3": "test"}

    def iter_samples(self) -> Iterator[Sample]:
        root = self.root
        img_dir = root / self.images_dir
        lbl_dir = root / self.labels_dir
        info_path = root / self.info_csv

        if not info_path.exists():
            raise ValueError(f"[Lizard] info.csv not found: {info_path}")

        # filename -> split_id + source
        info: Dict[str, Dict[str, str]] = {}
        with info_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                fn = row["Filename"]
                info[fn] = row

        for fn, row in sorted(info.items()):
            # images named like dpath_1.png, glas_1.png, ...
            ip = img_dir / f"{fn}.png"
            if not ip.exists():
                # try raw
                ip = img_dir / fn
            if not ip.exists():
                continue

            ap = lbl_dir / f"{fn}.mat"
            if not ap.exists():
                continue

            split_raw = str(row.get("Split", ""))
            sp = self.split_map.get(split_raw, "unspecified")
            H, W = read_hw_fast(str(ip))
            yield Sample(
                dataset=self.dataset_name,
                sample_id=fn,
                split=sp,  # type: ignore
                image_path=ip,
                ann_path=ap,
                group_id=row.get("Source") or fn,
                meta={
                    "split_raw": split_raw, 
                    "source": row.get("Source"),
                    "height_px": H,
                    "width_px": W,
                    "mpp_x": 0.5,
                    "mpp_y": 0.5,
                    },
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        assert sample.ann_path is not None
        mat = sio.loadmat(sample.ann_path)

        if "inst_map" not in mat:
            raise ValueError(f"[Lizard] inst_map not in {sample.ann_path}. keys={list(mat.keys())[:20]}")
        inst_map = mat["inst_map"].astype(np.int32)

        # class info (per nucleus) present in Lizard .mat
        nuclei_id = None
        classes = None
        if "id" in mat:
            nuclei_id = np.squeeze(mat["id"]).astype(np.int32)
        if "class" in mat:
            classes = np.squeeze(mat["class"]).astype(np.int32)

        id2cls: Dict[int, int] = {}
        if nuclei_id is not None and classes is not None and nuclei_id.size == classes.size:
            for i in range(int(nuclei_id.size)):
                id2cls[int(nuclei_id[i])] = int(classes[i])

        slices = find_instance_slices(inst_map)
        # for inst_id, slc in enumerate(slices, start=1):
        #     if slc is None:
        #         continue
        #     sub = inst_map[slc]
        #     m = (sub == inst_id)
        #     if not np.any(m):
        #         continue

        #     cls_id = id2cls.get(inst_id, 0)
        #     label = f"class_{cls_id}" if cls_id != 0 else "unknown"

        #     y0, y1 = slc[0].start, slc[0].stop
        #     x0, x1 = slc[1].start, slc[1].stop
        #     full = np.zeros(inst_map.shape, dtype=bool)
        #     full[y0:y1, x0:x1] = m
        H, W = inst_map.shape
        for inst_id, slc in enumerate(slices):
            if inst_id == 0 or slc is None:
                continue
            sub = inst_map[slc]
            m = (sub == inst_id)
            if not np.any(m):
                continue

            cls_id = id2cls.get(inst_id, 0)
            label = f"class_{cls_id}" if cls_id != 0 else "unknown"

            y0, y1 = slc[0].start, slc[0].stop
            x0, x1 = slc[1].start, slc[1].stop
            full = np.zeros((H, W), dtype=bool)
            full[y0:y1, x0:x1] = m

            yield AnnObject(
                ann_id=f"{sample.sample_id}:n{inst_id}",
                kind="instance",
                source_label=label,
                source_label_id=(cls_id if cls_id != 0 else None),
                mask=full,
                bbox_xywh=bbox_from_mask(full),
                area=area_from_mask(full),
            )
