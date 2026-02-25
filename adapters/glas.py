# dataflow/adapters/glas.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .types import AnnObject, BaseAdapter, Sample
from .utils import read_rgb, read_mask_any, find_instance_slices, bbox_from_mask, area_from_mask, read_hw_fast
from scipy import ndimage

class GlaSAdapter(BaseAdapter):
    dataset_name = "glas"

    def __init__(
        self,
        root: Path,
        *,
        pattern_map: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(root)
        # default: train_* -> train, testA_* -> val, testB_* -> test
        self.pattern_map = pattern_map or {"train_": "train", "testA_": "val", "testB_": "test"}

    def iter_samples(self) -> Iterator[Sample]:
        root = self.root
        # images and annos in one folder
        imgs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".bmp" and not p.stem.endswith("_anno")]
        if not imgs:
            raise ValueError(f"[GlaS] No .bmp images found under {root}")

        for ip in sorted(imgs):
            stem = ip.stem
            ap = root / f"{stem}_anno.bmp"
            if not ap.exists():
                # some distributions: anno has different suffix; you can extend here
                continue

            split = "unspecified"
            for pref, sp in self.pattern_map.items():
                if stem.startswith(pref):
                    split = sp
                    break
            H, W = read_hw_fast(str(ip))
            yield Sample(
                dataset=self.dataset_name,
                sample_id=stem,
                split=split,  # type: ignore
                image_path=ip,
                ann_path=ap,
                group_id=stem,  # can be replaced by patient id from Grades.csv later if needed
                meta={
                    "height_px": H,
                    "width_px": W,
                    "mpp_x": 0.620,
                    "mpp_y": 0.620,
                },
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        return read_rgb(sample.image_path)

    # def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
    #     assert sample.ann_path is not None
    #     ann = read_mask_any(sample.ann_path)

    #     # anno is often single-channel instance id map; if RGB, take channel 0 if identical
    #     if ann.ndim == 3:
    #         if np.all(ann[..., 0] == ann[..., 1]) and np.all(ann[..., 0] == ann[..., 2]):
    #             ann = ann[..., 0]
    #         else:
    #             ann = ann[..., 0]

    #     ann = ann.astype(np.int32)
        
    #     uniq = np.unique(ann)

    #     # 认为二值/伪二值：只有 0 和 1（或 0 和 255）等两种值
    #     if uniq.size <= 2:
    #         binary = ann > 0
    #         cc, n_cc = ndimage.label(binary)  # 1..n_cc instance ids
    #         for inst_id in range(1, n_cc + 1):
    #             m = (cc == inst_id)
    #             if not np.any(m):
    #                 continue
    #             yield AnnObject(
    #                 ann_id=f"{sample.sample_id}:cc{inst_id}",
    #                 kind="instance",
    #                 source_label="gland",
    #                 source_label_id=1,
    #                 mask=m,
    #                 bbox_xywh=bbox_from_mask(m),
    #                 area=area_from_mask(m),
    #             )
    #         return

    #     slices = find_instance_slices(ann)
    #     for idx, slc in enumerate(slices, start=1):
    #         if slc is None:
    #             continue
    #         sub = ann[slc]
    #         m = (sub == idx)
    #         if not np.any(m):
    #             continue

    #         y0, y1 = slc[0].start, slc[0].stop
    #         x0, x1 = slc[1].start, slc[1].stop
    #         full = np.zeros(ann.shape, dtype=bool)
    #         full[y0:y1, x0:x1] = m

    #         yield AnnObject(
    #             ann_id=f"{sample.sample_id}:inst{idx}",
    #             kind="instance",
    #             source_label="gland",
    #             source_label_id=1,
    #             mask=full,
    #             bbox_xywh=bbox_from_mask(full),
    #             area=area_from_mask(full),
    #         )
    def iter_ann(self, sample: Sample) -> Iterator[AnnObject]:
        assert sample.ann_path is not None
        # 1. 读取标注
        ann = read_mask_any(sample.ann_path)

        # 2. 统一转换为单通道 int32
        if ann.ndim == 3:
            # 即使是 RGB，也取第一个通道，但增加一个 check 确保万一
            ann = ann[..., 0]
        ann = ann.astype(np.int32)
        
        H, W = ann.shape
        uniq = np.unique(ann)
        # 过滤掉背景 (0)
        instance_ids = uniq[uniq > 0]

        if instance_ids.size == 0:
            return

        # 3. 检查是否为二值图（即需要手动连通域分割）
        # GlaS 官方数据集中 _anno.bmp 通常已经是 instance map (1, 2, 3...)
        # 但如果是某些预处理版本只有 0 和 255，则需要 label
        if instance_ids.size == 1 and (instance_ids[0] == 1 or instance_ids[0] == 255):
            binary = ann > 0
            cc, n_cc = ndimage.label(binary)
            instance_ids = np.arange(1, n_cc + 1)
            source_map = cc
        else:
            source_map = ann

        # 4. 获取每个实例的切片（加速提取）
        # 使用 find_objects 避免全图扫描
        slices = ndimage.find_objects(source_map)

        for inst_id in instance_ids:
            # find_objects 返回的 slice 索引是从 0 开始的，
            # 所以 ID 为 n 的物体在索引 n-1 处
            slc = slices[inst_id - 1]
            if slc is None:
                continue
                
            # 提取局部掩码
            sub_mask = (source_map[slc] == inst_id)
            if not np.any(sub_mask):
                continue

            # 计算 bbox (x, y, w, h)
            y_slice, x_slice = slc
            y0, x0 = y_slice.start, x_slice.start
            bh, bw = sub_mask.shape

            # 构造完整掩码（仅在必要时）
            # 如果你的 Parquet 逻辑支持直接处理 crop，这里可以优化
            # 目前根据你的 AnnObject 定义，仍需提供 full mask
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[slc] = sub_mask

            yield AnnObject(
                ann_id=f"{sample.sample_id}:inst{inst_id}",
                kind="instance",
                source_label="gland",
                source_label_id=1,
                mask=full_mask,
                bbox_xywh=(x0, y0, bw, bh), # 明确转换
                area=int(np.count_nonzero(sub_mask)), # 直接算局部更快
            )