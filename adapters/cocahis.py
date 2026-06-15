# dataflow/adapters/cocahis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Any

import h5py
import numpy as np

from .types import AnnObject, BaseAdapter, Sample
from .utils import bbox_from_mask, area_from_mask


def _decode_value(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return x.astype(str).item()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _read_group_field(group: h5py.Group, name: str) -> Optional[np.ndarray]:
    if name in group.attrs:
        return np.asarray(group.attrs[name])
    if name in group:
        return np.asarray(group[name])
    return None


def _normalize_split(x: Any) -> str:
    x = _decode_value(x)
    s = str(x).strip().lower()

    if s in {"train", "training"}:
        return "train"
    if s in {"test", "testing"}:
        return "test"
    if s in {"val", "valid", "validation"}:
        return "val"
    return "unspecified"


class CoCaHisAdapter(BaseAdapter):
    dataset_name = "cocahis"

    def __init__(
        self,
        root: Path,
        *,
        h5_relpath: str = "CoCaHis/CoCaHis.hdf5",
        image_key: str = "raw",  # "raw" | "sn1" | "sn2"
        gt_key: str = "GT_majority_vote",  # or "GT1" ... "GT7"
    ) -> None:
        super().__init__(root)

        self.h5_path = self.root / h5_relpath

        if not self.h5_path.exists():
            alt = self.root / "CoCaHis.hdf5"
            if alt.exists():
                self.h5_path = alt

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"[CoCaHis] Cannot find hdf5 file. Tried: "
                f"{self.root / h5_relpath} and {self.root / 'CoCaHis.hdf5'}"
            )

        self.image_key = image_key
        self.gt_key = gt_key

        self.id2name: Dict[int, str] = {
            1: "foreground",
            255: "foreground",
        }

    def iter_samples(self) -> Iterator[Sample]:
        with h5py.File(self.h5_path, "r") as f:
            if "HE" not in f or "GT" not in f:
                raise ValueError(
                    f"[CoCaHis] Expected top-level groups 'HE' and 'GT' in {self.h5_path}. "
                    f"Available top-level keys: {list(f.keys())}"
                )

            he_group = f["HE"]
            gt_group = f["GT"]

            if self.image_key not in he_group:
                raise ValueError(
                    f"[CoCaHis] HE/{self.image_key} not found. "
                    f"Available HE keys: {list(he_group.keys())}"
                )

            if self.gt_key not in gt_group:
                raise ValueError(
                    f"[CoCaHis] GT/{self.gt_key} not found. "
                    f"Available GT keys: {list(gt_group.keys())}"
                )

            img_ds = he_group[self.image_key]
            gt_ds = gt_group[self.gt_key]

            if img_ds.shape[0] != gt_ds.shape[0]:
                raise ValueError(
                    f"[CoCaHis] Image/mask count mismatch: "
                    f"HE/{self.image_key} has {img_ds.shape[0]}, "
                    f"GT/{self.gt_key} has {gt_ds.shape[0]}"
                )

            n = int(img_ds.shape[0])
            H = int(img_ds.shape[1])
            W = int(img_ds.shape[2])

            split_arr = _read_group_field(he_group, "train_test_split")
            if split_arr is None:
                split_arr = _read_group_field(gt_group, "train_test_split")

            patient_arr = _read_group_field(he_group, "patient_num")
            image_num_arr = _read_group_field(he_group, "image_num")

            for i in range(n):
                split = (
                    _normalize_split(split_arr[i])
                    if split_arr is not None and len(split_arr) > i
                    else "unspecified"
                )

                patient_num = (
                    int(_decode_value(patient_arr[i]))
                    if patient_arr is not None and len(patient_arr) > i
                    else None
                )

                image_num = (
                    int(_decode_value(image_num_arr[i]))
                    if image_num_arr is not None and len(image_num_arr) > i
                    else i
                )

                if patient_num is not None:
                    sample_id = f"p{patient_num:02d}_img{image_num:03d}"
                    group_id = f"patient_{patient_num:02d}"
                else:
                    sample_id = f"idx{i:03d}"
                    group_id = sample_id

                yield Sample(
                    dataset=self.dataset_name,
                    sample_id=sample_id,
                    split=split,  # type: ignore
                    image_path=self.h5_path,
                    ann_path=self.h5_path,
                    group_id=group_id,
                    meta={
                        "h5_path": str(self.h5_path),
                        "h5_index": i,
                        "image_key": self.image_key,
                        "gt_key": self.gt_key,
                        "patient_num": patient_num,
                        "image_num": image_num,
                        "height_px": H,
                        "width_px": W,
                        "label_map": dict(self.id2name),
                    },
                )

    def load_image(self, sample: Sample) -> np.ndarray:
        idx = int(sample.meta["h5_index"])
        image_key = str(sample.meta.get("image_key", self.image_key))

        with h5py.File(self.h5_path, "r") as f:
            arr = np.asarray(f["HE"][image_key][idx])

        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"[CoCaHis] Expected RGB image HxWx3, got shape {arr.shape}")

        return arr.astype(np.uint8, copy=False)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        idx = int(sample.meta["h5_index"])
        gt_key = str(sample.meta.get("gt_key", self.gt_key))

        with h5py.File(self.h5_path, "r") as f:
            mask_arr = np.asarray(f["GT"][gt_key][idx])

        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]

        uniq = [int(x) for x in np.unique(mask_arr).tolist()]
        fg_ids = [x for x in uniq if x != 0]

        if not fg_ids:
            return

        if len(fg_ids) == 1:
            cid = fg_ids[0]
            m = mask_arr > 0
            yield AnnObject(
                ann_id=f"{sample.sample_id}:{gt_key}:fg",
                kind="semantic",
                source_label=self.id2name.get(cid, "foreground"),
                source_label_id=cid,
                mask=m,
                bbox_xywh=bbox_from_mask(m),
                area=area_from_mask(m),
            )
            return

        for cid in fg_ids:
            m = mask_arr == cid
            yield AnnObject(
                ann_id=f"{sample.sample_id}:{gt_key}:c{cid}",
                kind="semantic",
                source_label=self.id2name.get(cid, f"class_{cid}"),
                source_label_id=cid,
                mask=m,
                bbox_xywh=bbox_from_mask(m),
                area=area_from_mask(m),
            )