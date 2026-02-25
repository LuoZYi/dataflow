# dataflow/adapters/utils.py
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

@lru_cache(maxsize=200_000)
def read_hw_fast(path_str: str) -> tuple[int, int]:
    p = Path(path_str)
    with Image.open(p) as im:
        w, h = im.size
    return int(h), int(w)  # (H, W)

def list_images(dir_path: Path, exts: Optional[Sequence[str]] = None) -> List[Path]:
    exts_set = {e.lower() for e in (exts or IMG_EXTS)}
    out: List[Path] = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts_set:
            out.append(p)
    return sorted(out)


def index_by_stem(dir_path: Path, exts: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in list_images(dir_path, exts=exts):
        s = p.stem
        if s in m:
            raise ValueError(f"Duplicate stem under {dir_path}: {s}\n  {m[s]}\n  {p}")
        m[s] = p
    return m


def read_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def read_mask_any(path: Path) -> np.ndarray:
    """
    Read mask png/bmp/tif etc.
    Returns:
      - (H,W) uint8/uint16 if grayscale/palette
      - (H,W,3) uint8 if RGB
    """
    img = Image.open(path)
    arr = np.asarray(img)
    return arr


def rgb_to_int(arr_rgb: np.ndarray) -> np.ndarray:
    """Pack RGB uint8 to int32: r<<16 + g<<8 + b."""
    r = arr_rgb[..., 0].astype(np.int32)
    g = arr_rgb[..., 1].astype(np.int32)
    b = arr_rgb[..., 2].astype(np.int32)
    return (r << 16) + (g << 8) + b


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def area_from_mask(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def find_instance_slices(inst_map: np.ndarray):
    """
    Efficiently get bounding slices for each integer label using scipy.ndimage.find_objects.
    Returns list indexed by label-1 (label starts from 1).
    """
    from scipy import ndimage  # lazy import
    return ndimage.find_objects(inst_map)


def mask_to_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    return mask.astype(bool)
