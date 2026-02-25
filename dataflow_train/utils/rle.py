# dataflow_train/utils/rle.py
from __future__ import annotations
from typing import Any, List
import numpy as np


def _to_counts_list(counts: Any) -> List[int]:
    if counts is None:
        return []
    if isinstance(counts, (list, tuple)):
        return [int(x) for x in counts]
    if isinstance(counts, np.ndarray):
        return [int(x) for x in counts.tolist()]
    if isinstance(counts, str):
        s = counts.strip()
        if s.startswith("[") and s.endswith("]"):
            import json
            try:
                return [int(x) for x in json.loads(s)]
            except Exception:
                pass
        return [int(x) for x in s.replace(",", " ").split()]
    return [int(counts)]


def rle_decode_counts(counts: Any, h: int, w: int) -> np.ndarray:
    counts = _to_counts_list(counts)
    total = int(h) * int(w)
    if not counts:
        return np.zeros((h, w), dtype=bool)
    if int(sum(counts)) != total:
        raise ValueError(f"sum(counts)={sum(counts)} != h*w={total} (h={h},w={w})")

    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        run = int(run)
        if val == 1 and run > 0:
            flat[idx:idx + run] = 1
        idx += run
        val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)
