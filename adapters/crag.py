# # dataflow/adapters/crag.py
# from __future__ import annotations

# import json
# import re
# from pathlib import Path
# from typing import Dict, Iterator, List, Optional, Tuple

# import numpy as np
# from PIL import Image, ImageDraw

# from .types import AnnObject, BaseAdapter, Sample
# from .utils import read_rgb, bbox_from_mask, area_from_mask, read_hw_fast


# def _load_coco(path: Path) -> Dict:
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def _rasterize_polygons(
#     polys: List[List[float]],
#     h: int,
#     w: int,
# ) -> np.ndarray:
#     """
#     polys: list of flattened [x1,y1,x2,y2,...] (COCO format)
#     """
#     mask_img = Image.new("L", (w, h), 0)
#     draw = ImageDraw.Draw(mask_img)
#     for poly in polys:
#         if len(poly) < 6:
#             continue
#         pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
#         draw.polygon(pts, outline=1, fill=1)
#     return np.asarray(mask_img).astype(bool)


# class CRAGAdapter(BaseAdapter):
#     dataset_name = "crag"

#     def __init__(
#         self,
#         root: Path,
#         *,
#         include_aug: bool = True,
#         aug_regex: str = r"_aug_\d+$",
#     ) -> None:
#         super().__init__(root)
#         self.include_aug = include_aug
#         self.aug_re = re.compile(aug_regex)

#     def iter_samples(self) -> Iterator[Sample]:
#         root = self.root
#         ann_dir = root / "annotations"
#         train_json = ann_dir / "instances_train2017.json"
#         val_json = ann_dir / "instances_val2017.json"
#         if not train_json.exists() or not val_json.exists():
#             raise ValueError(f"[CRAG] Cannot find jsons under {ann_dir}")

#         for split_name, img_dir, coco_path in [
#             ("train", root / "train2017", train_json),
#             ("val", root / "val2017", val_json),
#         ]:
#             coco = _load_coco(coco_path)
#             # build image_id -> file_name
#             id2img: Dict[int, Dict] = {int(x["id"]): x for x in coco.get("images", [])}
#             for img_id, info in id2img.items():
#                 fn = info.get("file_name") or info.get("name")
#                 if fn is None:
#                     continue
#                 stem = Path(fn).stem
#                 if (not self.include_aug) and self.aug_re.search(stem):
#                     continue
#                 ip = img_dir / fn
#                 if not ip.exists():
#                     # some exports store only basename
#                     ip = img_dir / f"{stem}{Path(fn).suffix or '.png'}"
#                 if not ip.exists():
#                     continue
#                 H, W = read_hw_fast(str(ip))  
#                 yield Sample(
#                     dataset=self.dataset_name,
#                     sample_id=f"{split_name}:{stem}",
#                     split=split_name,  # type: ignore
#                     image_path=ip,
#                     ann_path=coco_path,
#                     group_id=stem.split("_aug_")[0],
#                     meta={
#                         "image_id": img_id,
#                         "file_name": fn,
#                         "height_px": H, 
#                         "width_px": W,
#                         "mpp_x": 0.5,
#                         "mpp_y": 0.5,
#                     },
#                 )

#     def load_image(self, sample: Sample) -> np.ndarray:
#         return read_rgb(sample.image_path)

#     def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
#         assert sample.ann_path is not None
#         coco = _load_coco(sample.ann_path)
#         img_id = int(sample.meta["image_id"])

#         # collect anns for this image
#         anns = [a for a in coco.get("annotations", []) if int(a.get("image_id", -1)) == img_id]

#         # need H,W for rasterization
#         # prefer meta, else read image
#         h = sample.meta.get("height")
#         w = sample.meta.get("width")
#         if h is None or w is None:
#             arr = self.load_image(sample)
#             h, w = arr.shape[0], arr.shape[1]
#         h = int(h)
#         w = int(w)

#         for a in anns:
#             seg = a.get("segmentation", None)
#             if not seg:
#                 continue
#             # COCO polygon: segmentation is list of polygons
#             polys: List[List[float]] = [list(map(float, p)) for p in seg]
#             mask = _rasterize_polygons(polys, h=h, w=w)

#             cid = int(a.get("category_id", 1))
#             label = "gland" if cid == 1 else f"class_{cid}"

#             yield AnnObject(
#                 ann_id=f"{sample.sample_id}:ann{int(a.get('id', 0))}",
#                 kind="instance",
#                 source_label=label,
#                 source_label_id=cid,
#                 mask=mask,
#                 polygons=polys,  # keep both; parquet writer can choose one
#                 bbox_xywh=bbox_from_mask(mask),
#                 area=area_from_mask(mask),
#                 meta={"iscrowd": int(a.get("iscrowd", 0))},
#             )


# dataflow/adapters/crag.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .types import AnnObject, BaseAdapter, Sample
from .utils import read_rgb, bbox_from_mask, area_from_mask, read_hw_fast


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_coco_jsons(ann_dir: Path) -> Tuple[Path, Path]:
    """
    CRAG 常见两种命名：
      - instances_train2017.json / instances_val2017.json
      - instance_train2017.json  / instance_val2017.json
    """
    candidates = [
        ("instances_train2017.json", "instances_val2017.json"),
        ("instance_train2017.json", "instance_val2017.json"),
    ]
    for a, b in candidates:
        p1, p2 = ann_dir / a, ann_dir / b
        if p1.exists() and p2.exists():
            return p1, p2
    raise ValueError(f"[CRAG] Cannot find train/val json under {ann_dir}. Tried: {candidates}")


def _rasterize_polygons(polys: List[List[float]], h: int, w: int) -> np.ndarray:
    """
    polys: list of flattened [x1,y1,x2,y2,...] (COCO polygon format)
    """
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polys:
        if len(poly) < 6:
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        # PIL 会自动裁剪出界点；这里不强制 clamp（保持简单）
        draw.polygon(pts, outline=1, fill=1)

    return np.asarray(mask_img).astype(bool)


class CRAGAdapter(BaseAdapter):
    dataset_name = "crag"

    def __init__(
        self,
        root: Path,
        *,
        include_aug: bool = True,
        aug_regex: str = r"_aug_\d+$",
        default_mpp: float = 0.5,
    ) -> None:
        super().__init__(root)
        self.include_aug = include_aug
        self.aug_re = re.compile(aug_regex)
        self.default_mpp = float(default_mpp)

        # cache
        self._coco_cache: Dict[Path, Dict] = {}
        # index cache: coco_path -> (id2img, imgid2anns)
        self._index_cache: Dict[Path, Tuple[Dict[int, Dict], Dict[int, List[Dict]]]] = {}

    def _get_coco(self, coco_path: Path) -> Dict:
        coco = self._coco_cache.get(coco_path)
        if coco is None:
            coco = _load_json(coco_path)
            self._coco_cache[coco_path] = coco
        return coco

    def _get_index(self, coco_path: Path) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]]]:
        idx = self._index_cache.get(coco_path)
        if idx is not None:
            return idx

        coco = self._get_coco(coco_path)
        id2img: Dict[int, Dict] = {int(x["id"]): x for x in coco.get("images", [])}

        imgid2anns: Dict[int, List[Dict]] = {}
        for a in coco.get("annotations", []):
            iid = int(a.get("image_id", -1))
            if iid < 0:
                continue
            imgid2anns.setdefault(iid, []).append(a)

        self._index_cache[coco_path] = (id2img, imgid2anns)
        return self._index_cache[coco_path]

    def iter_samples(self) -> Iterator[Sample]:
        root = self.root
        ann_dir = root / "annotations"
        train_json, val_json = _find_coco_jsons(ann_dir)

        splits = [
            ("train", root / "train2017", train_json),
            ("val", root / "val2017", val_json),
        ]

        for split_name, img_dir, coco_path in splits:
            id2img, _ = self._get_index(coco_path)

            for img_id, info in id2img.items():
                fn = info.get("file_name") or info.get("name")
                if not fn:
                    continue
                stem = Path(fn).stem
                if (not self.include_aug) and self.aug_re.search(stem):
                    continue

                ip = img_dir / fn
                if not ip.exists():
                    # fallback: only basename exists
                    ip = img_dir / Path(fn).name
                if not ip.exists():
                    continue

                # 尽量不用开图：优先用 COCO height/width
                H = info.get("height")
                W = info.get("width")
                if H is None or W is None:
                    h2, w2 = read_hw_fast(str(ip))
                    H, W = H or h2, W or w2

                yield Sample(
                    dataset=self.dataset_name,
                    sample_id=f"{split_name}:{stem}",
                    split=split_name,  # type: ignore
                    image_path=ip,
                    ann_path=coco_path,
                    group_id=stem.split("_aug_")[0],
                    meta={
                        "image_id": int(img_id),
                        "file_name": str(fn),
                        "height_px": int(H),
                        "width_px": int(W),
                        "mpp_x": self.default_mpp,
                        "mpp_y": self.default_mpp,
                    },
                )

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        assert sample.ann_path is not None
        img_id = int(sample.meta["image_id"])

        _, imgid2anns = self._get_index(sample.ann_path)
        anns = imgid2anns.get(img_id, [])

        # H/W 统一读 height_px/width_px
        h = sample.meta.get("height_px")
        w = sample.meta.get("width_px")
        if h is None or w is None:
            # 兜底：header 读一下（不建议 load_image）
            h2, w2 = read_hw_fast(str(sample.image_path))
            h, w = h or h2, w or w2
        h = int(h)
        w = int(w)

        for a in anns:
            seg = a.get("segmentation", None)
            if not seg:
                continue

            # COCO polygon: segmentation is list of polygons
            polys: List[List[float]] = [list(map(float, p)) for p in seg]
            mask = _rasterize_polygons(polys, h=h, w=w)

            cid = int(a.get("category_id", 1))
            label = "gland" if cid == 1 else f"class_{cid}"

            yield AnnObject(
                ann_id=f"{sample.sample_id}:ann{int(a.get('id', 0))}",
                kind="instance",
                source_label=label,
                source_label_id=cid,
                mask=mask,
                polygons=polys,  # 可选：保留 polygon，后面 parquet writer 决定用哪个
                bbox_xywh=bbox_from_mask(mask),
                area=area_from_mask(mask),
                meta={"iscrowd": int(a.get("iscrowd", 0))},
            )
