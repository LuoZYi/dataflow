# dataflow/adapters/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Literal

SplitName = Literal["train", "val", "test", "unspecified"]
AnnKind = Literal["instance", "semantic"]  # semantic: one region per class (still treated as an object)


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str               # unique within dataset
    split: SplitName
    image_path: Path
    ann_path: Optional[Path]     # may be None for some formats (but here all should have)
    group_id: Optional[str] = None  # to prevent leakage; can be patient/slide id etc.
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnObject:
    """
    Unified in-memory annotation object.
    generate_parquet later can decide to store polygons, RLE, etc.
    """
    ann_id: str                  # unique within sample
    kind: AnnKind
    source_label: str            # dataset-specific label name (or "class_3")
    source_label_id: Optional[int] = None

    # geometry (one of these)
    mask: Optional["Any"] = None  # np.ndarray[H,W] bool/uint8 (kept Any to avoid hard numpy import here)
    polygons: Optional[List[List[float]]] = None  # COCO-style poly list (flattened)

    # cached stats (optional)
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None
    area: Optional[int] = None

    meta: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter:
    dataset_name: str

    def __init__(self, root: Path, **kwargs: Any) -> None:
        self.root = Path(root)

    def iter_samples(self) -> Iterator[Sample]:
        raise NotImplementedError

    def load_image(self, sample: Sample):
        raise NotImplementedError

    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        raise NotImplementedError
