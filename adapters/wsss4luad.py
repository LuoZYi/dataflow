# dataflow/adapters/wsss4luad.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .types import AnnObject, BaseAdapter, Sample
from .utils import read_rgb, read_mask_any, bbox_from_mask, area_from_mask, read_hw_fast


def _is_junk_file(p: Path) -> bool:
    name = p.name
    return (
        name.startswith(".")
        or name.startswith("._")
        or name == ".DS_Store"
        or name.endswith("~")
    )


def _list_pngs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*.png")
        if p.is_file() and not _is_junk_file(p)
    )


def _strip_train_label_suffix(stem: str) -> str:
    """
    Training filenames may look like:
      xxx-[1, 0, 0].png

    Return:
      xxx
    """
    return re.sub(r"[-_]*\[[0-9,\s]+\]$", "", stem).strip("-_")


def _parse_image_level_label(stem: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse final [Tumor, Stroma, Normal] multi-label vector from training filename.

    Example:
      xxx-[1, 0, 0].png -> (1, 0, 0)
    """
    m = re.search(r"\[([0-9,\s]+)\]$", stem)
    if not m:
        return None

    vals = [int(x.strip()) for x in m.group(1).split(",") if x.strip() != ""]
    if len(vals) != 3:
        return None

    return tuple(vals)  # type: ignore


def _norm_match_stem(stem: str) -> str:
    """
    Normalize image/mask names for matching.
    """
    s = _strip_train_label_suffix(stem)
    s = s.lower()

    suffixes = [
        "_mask", "-mask",
        "_masks", "-masks",
        "_label", "-label",
        "_labels", "-labels",
        "_gt", "-gt",
        "_gtruth", "-gtruth",
        "_annotation", "-annotation",
        "_annotations", "-annotations",
        "_seg", "-seg",
        "_segmentation", "-segmentation",
    ]

    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True

    return s.strip("-_.")


def _index_validation_or_test(
    split_root: Path,
) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path]]:
    """
    Explicit WSSS4LUAD validation/test structure.

    Expected:

    2.validation/
      img/
      mask/              semantic masks
      background-mask/   background/ignore masks, not semantic GT

    3.testing/
      img/
      background-mask/   background/ignore masks only

    Returns:
      img_map: image stem -> image path
      sem_map: image stem -> semantic mask path
      bg_map: image stem -> background mask path
    """
    img_dir = split_root / "img"
    sem_mask_dir = split_root / "mask"
    bg_mask_dir = split_root / "background-mask"

    img_map: Dict[str, Path] = {}
    sem_map: Dict[str, Path] = {}
    bg_map: Dict[str, Path] = {}

    for p in _list_pngs(img_dir):
        img_map[_norm_match_stem(p.stem)] = p

    for p in _list_pngs(sem_mask_dir):
        sem_map[_norm_match_stem(p.stem)] = p

    for p in _list_pngs(bg_mask_dir):
        bg_map[_norm_match_stem(p.stem)] = p

    return img_map, sem_map, bg_map


class WSSS4LUADAdapter(BaseAdapter):
    """
    WSSS4LUAD adapter.

    Important:
    - 1.training has image-level labels encoded in filenames.
      It does NOT have pixel-level masks.
    - 2.validation has semantic masks under mask/.
    - background-mask/ is stored in sample.meta only and is NOT yielded as AnnObject.
    - 3.testing has images and background masks, but no semantic masks.
    """

    dataset_name = "wsss4luad"

    def __init__(
        self,
        root: Path,
        *,
        training_dir: str = "1.training",
        validation_dir: str = "2.validation",
        testing_dir: str = "3.testing",
        include_train: bool = True,
        include_val: bool = True,
        include_test: bool = True,
    ) -> None:
        super().__init__(root)

        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.testing_dir = testing_dir

        self.include_train = include_train
        self.include_val = include_val
        self.include_test = include_test

        # WSSS4LUAD image-level label order is commonly:
        # [Tumor, Stroma, Normal]
        #
        # Validation semantic masks use:
        #   0 = background / ignore
        #   1 = tumor
        #   2 = stroma
        #   3 = normal
        self.id2name: Dict[int, str] = {
            1: "tumor",
            2: "stroma",
            3: "normal",
        }

    def _iter_training_samples(self) -> Iterator[Sample]:
        train_root = self.root / self.training_dir
        img_paths = _list_pngs(train_root)

        for img_path in img_paths:
            raw_stem = img_path.stem
            sample_id = _strip_train_label_suffix(raw_stem)
            image_level_label = _parse_image_level_label(raw_stem)

            H, W = read_hw_fast(str(img_path))

            meta = {
                "height_px": H,
                "width_px": W,
                "label_map": dict(self.id2name),
                "has_pixel_annotation": False,
                "has_background_mask": False,
                "background_mask_path": None,
                "supervision": "image_level",
            }

            if image_level_label is not None:
                t, s, n = image_level_label
                meta.update(
                    {
                        "image_level_label": {
                            "tumor": int(t),
                            "stroma": int(s),
                            "normal": int(n),
                        },
                        "image_level_label_vector": [int(t), int(s), int(n)],
                        "image_level_label_order": ["tumor", "stroma", "normal"],
                    }
                )

            yield Sample(
                dataset=self.dataset_name,
                sample_id=sample_id,
                split="train",
                image_path=img_path,
                ann_path=None,
                group_id=sample_id,
                meta=meta,
            )

    def _iter_eval_split_samples(self, split_name: str, split_dir: str) -> Iterator[Sample]:
        split_root = self.root / split_dir
        img_map, sem_map, bg_map = _index_validation_or_test(split_root)

        if not img_map:
            print(f"[WSSS4LUAD] WARNING: no images found in {split_root / 'img'}")

        for key in sorted(img_map.keys()):
            img_path = img_map[key]
            sem_path = sem_map.get(key)
            bg_path = bg_map.get(key)

            H, W = read_hw_fast(str(img_path))

            if sem_path is not None:
                supervision = "pixel_level"
            elif bg_path is not None:
                supervision = "image_only_with_background_mask"
            else:
                supervision = "image_only"

            yield Sample(
                dataset=self.dataset_name,
                sample_id=key,
                split=split_name,  # type: ignore
                image_path=img_path,
                ann_path=sem_path,
                group_id=key,
                meta={
                    "height_px": H,
                    "width_px": W,
                    "label_map": dict(self.id2name),
                    "has_pixel_annotation": sem_path is not None,
                    "has_background_mask": bg_path is not None,
                    "background_mask_path": str(bg_path) if bg_path is not None else None,
                    "supervision": supervision,
                },
            )

    def iter_samples(self) -> Iterator[Sample]:
        if self.include_train:
            yield from self._iter_training_samples()

        if self.include_val:
            yield from self._iter_eval_split_samples("val", self.validation_dir)

        if self.include_test:
            yield from self._iter_eval_split_samples("test", self.testing_dir)

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        # Training samples and testing samples may have no semantic annotation.
        if sample.ann_path is None:
            return

        mask_arr = read_mask_any(sample.ann_path)

        # If RGB but actually grayscale repeated channels, collapse to 2D.
        if mask_arr.ndim == 3:
            if (
                mask_arr.shape[-1] >= 3
                and np.all(mask_arr[..., 0] == mask_arr[..., 1])
                and np.all(mask_arr[..., 0] == mask_arr[..., 2])
            ):
                mask_arr = mask_arr[..., 0]
            else:
                raise ValueError(
                    f"[WSSS4LUAD] Expected indexed/grayscale semantic mask, "
                    f"but got RGB mask for {sample.ann_path} with shape {mask_arr.shape}"
                )

        uniq_ids = [int(x) for x in np.unique(mask_arr).tolist() if int(x) != 0]

        for cid in uniq_ids:
            m = mask_arr == cid
            name = sample.meta.get("label_map", {}).get(cid, f"class_{cid}")

            yield AnnObject(
                ann_id=f"{sample.sample_id}:c{cid}",
                kind="semantic",
                source_label=str(name),
                source_label_id=cid,
                mask=m,
                bbox_xywh=bbox_from_mask(m),
                area=area_from_mask(m),
                meta={"mask_value": int(cid)},
            )