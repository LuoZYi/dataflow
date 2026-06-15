from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .types import AnnObject, BaseAdapter, Sample
from .utils import read_rgb, read_mask_any, bbox_from_mask, area_from_mask, read_hw_fast


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _is_junk_file(p: Path) -> bool:
    return (
        p.name.startswith(".")
        or p.name.startswith("._")
        or p.name == ".DS_Store"
        or p.name.endswith("~")
    )


def _is_image_file(p: Path) -> bool:
    return p.is_file() and not _is_junk_file(p) and p.suffix.lower() in IMG_EXTS


def _strip_pair_suffix(stem: str) -> str:
    for suf in [
        "_HE", "-HE", "_he", "-he",
        "_mask", "-mask", "_Mask", "-Mask", "_MASK", "-MASK",
    ]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _mask_to_binary(mask_arr: np.ndarray) -> np.ndarray:
    """
    SegPath masks can visually look black because foreground may be value 1.
    Treat any non-zero pixel as foreground.
    """
    if mask_arr.ndim == 3:
        if (
            mask_arr.shape[-1] >= 3
            and np.all(mask_arr[..., 0] == mask_arr[..., 1])
            and np.all(mask_arr[..., 0] == mask_arr[..., 2])
        ):
            mask_arr = mask_arr[..., 0]
        else:
            mask_arr = np.any(mask_arr[..., :3] > 0, axis=-1).astype(np.uint8)

    return mask_arr > 0


def _find_image_mask_dirs(root: Path) -> List[Path]:
    """
    Find every directory under root that directly contains images/ and masks/.

    Expected cleaned structure:
      SegPath_clean/
        endothelial_cells/
          images/
          masks/
        leukocytes/
          images/
          masks/
        ...
    """
    out: List[Path] = []

    all_dirs = [root]
    all_dirs.extend(sorted(p for p in root.rglob("*") if p.is_dir()))

    for d in all_dirs:
        if (d / "images").is_dir() and (d / "masks").is_dir():
            out.append(d)

    return sorted(out)


def _label_from_prefix(image_path: Path) -> str:
    """
    Infer antibody/concept label from filename.

    Example:
      ERG_Endothelium_024_067584_052224_HE.png -> ERG_Endothelium
      CD45RB_Leukocyte_..._HE.png              -> CD45RB_Leukocyte
      panCK_Epithelium_..._HE.png              -> panCK_Epithelium
    """
    stem = _strip_pair_suffix(image_path.stem)
    parts = stem.split("_")

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return parts[0] if parts else image_path.parent.parent.name


class SegPathCleanAdapter(BaseAdapter):
    """
    Adapter for cleaned SegPath structure.

    Expected:
      root/
        concept_or_cell_type/
          images/
            xxx_HE.png
          masks/
            xxx_mask.png

    Design choice:
      - orphan image/mask should already be cleaned before this adapter.
      - empty masks are skipped entirely.
      - one non-empty binary mask = one semantic annotation.
      - image samples with empty masks are NOT yielded.
    """

    dataset_name = "segpath"

    def __init__(
        self,
        root: Path,
        *,
        min_area: int = 1,
        label_mode: str = "prefix",  # "prefix" | "leaf" | "relative"
        split: str = "unspecified",
    ) -> None:
        super().__init__(root)
        self.min_area = int(min_area)
        self.label_mode = str(label_mode)
        self.default_split = str(split)

        self.name2id: Dict[str, int] = {}
        self.id2name: Dict[int, str] = {}

        self._records: Optional[List[Dict[str, object]]] = None
        self._last_discovery_debug: Dict[str, object] = {}

    def _label_name(self, dataset_dir: Path, image_path: Path) -> str:
        if self.label_mode == "relative":
            return dataset_dir.relative_to(self.root).as_posix().replace("/", "__")

        if self.label_mode == "leaf":
            return dataset_dir.name

        if self.label_mode == "prefix":
            return _label_from_prefix(image_path)

        raise ValueError(f"[SegPathClean] Unknown label_mode={self.label_mode}")

    def _label_id(self, name: str) -> int:
        if name not in self.name2id:
            cid = len(self.name2id) + 1
            self.name2id[name] = cid
            self.id2name[cid] = name
        return self.name2id[name]

    def _discover_records(self) -> List[Dict[str, object]]:
        if self._records is not None:
            return self._records

        dataset_dirs = _find_image_mask_dirs(self.root)

        records: List[Dict[str, object]] = []

        total_images = 0
        total_masks = 0
        paired_raw = 0
        kept_nonempty = 0
        skipped_empty = 0
        skipped_small = 0
        orphan_images = 0
        orphan_masks = 0

        for d in dataset_dirs:
            image_dir = d / "images"
            mask_dir = d / "masks"

            image_files = sorted(p for p in image_dir.rglob("*") if _is_image_file(p))
            mask_files = sorted(p for p in mask_dir.rglob("*") if _is_image_file(p))

            total_images += len(image_files)
            total_masks += len(mask_files)

            image_map = {_strip_pair_suffix(p.stem): p for p in image_files}
            mask_map = {_strip_pair_suffix(p.stem): p for p in mask_files}

            image_keys = set(image_map.keys())
            mask_keys = set(mask_map.keys())

            orphan_images += len(image_keys - mask_keys)
            orphan_masks += len(mask_keys - image_keys)

            paired_keys = sorted(image_keys & mask_keys)

            for key in paired_keys:
                img = image_map[key]
                mask = mask_map[key]
                paired_raw += 1

                mask_arr = read_mask_any(mask)
                binary = _mask_to_binary(mask_arr)
                area = area_from_mask(binary)

                if area <= 0:
                    skipped_empty += 1
                    continue

                if area < self.min_area:
                    skipped_small += 1
                    continue

                label_name = self._label_name(d, img)
                label_id = self._label_id(label_name)

                rel_key = f"{d.relative_to(self.root).as_posix()}/{key}"

                records.append({
                    "key": rel_key,
                    "image_path": img,
                    "mask_path": mask,
                    "label_name": label_name,
                    "label_id": label_id,
                    "subdataset_dir": d,
                    "mask_area": int(area),
                })
                kept_nonempty += 1

        self._last_discovery_debug = {
            "root": str(self.root),
            "label_mode": self.label_mode,
            "min_area": self.min_area,
            "dataset_dirs": [str(d) for d in dataset_dirs],
            "num_dataset_dirs": len(dataset_dirs),
            "total_images": total_images,
            "total_masks": total_masks,
            "paired_raw": paired_raw,
            "kept_nonempty": kept_nonempty,
            "skipped_empty": skipped_empty,
            "skipped_small": skipped_small,
            "orphan_images": orphan_images,
            "orphan_masks": orphan_masks,
            "records": len(records),
            "id2name": dict(self.id2name),
        }

        self._records = records
        return records

    def iter_samples(self) -> Iterator[Sample]:
        records = self._discover_records()

        if not records:
            raise RuntimeError(
                f"[SegPathClean] No non-empty paired image/mask records found under {self.root}"
            )

        for r in records:
            image_path = Path(r["image_path"])
            mask_path = Path(r["mask_path"])

            H, W = read_hw_fast(str(image_path))

            sample_id = str(r["key"]).replace("/", ":")
            group_id = str(r["key"])

            yield Sample(
                dataset=self.dataset_name,
                sample_id=sample_id,
                split=self.default_split,  # type: ignore
                image_path=image_path,
                ann_path=mask_path,
                group_id=group_id,
                meta={
                    "height_px": H,
                    "width_px": W,
                    "label_map": dict(self.id2name),
                    "label_name": str(r["label_name"]),
                    "label_id": int(r["label_id"]),
                    "subdataset_dir": str(r["subdataset_dir"]),
                    "mask_path": str(mask_path),
                    "mask_area": int(r["mask_area"]),
                },
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        assert sample.ann_path is not None

        mask_arr = read_mask_any(sample.ann_path)
        mask = _mask_to_binary(mask_arr)

        area = area_from_mask(mask)
        if area < self.min_area:
            return

        cid = int(sample.meta["label_id"])
        name = str(sample.meta["label_name"])

        yield AnnObject(
            ann_id=f"{sample.sample_id}:c{cid}",
            kind="semantic",
            source_label=name,
            source_label_id=cid,
            mask=mask,
            bbox_xywh=bbox_from_mask(mask),
            area=area,
            meta={
                "mask_path": str(sample.ann_path),
                "subdataset_dir": sample.meta.get("subdataset_dir"),
                "mask_mode": "binary_nonzero",
            },
        )
