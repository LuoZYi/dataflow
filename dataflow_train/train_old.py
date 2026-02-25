# # dataflow_train/train.py
# import os, sys, random, logging, traceback
# from argparse import ArgumentParser
# from pathlib import Path

# import torch
# import submitit
# from hydra import compose, initialize_config_module
# from hydra.utils import instantiate
# from omegaconf import OmegaConf
# import pandas as pd
# import numpy as np
# from pathlib import Path

# db = Path("/home/path_sam3/dataflow/parquet_db/v1")
# ds = "crag"
# meta = pd.read_parquet(db/ds/"meta.parquet", columns=["slide_uid","height_px","width_px"])
# ann  = pd.read_parquet(db/ds/"ann_semantic.parquet", columns=["slide_uid","area"])


# class Tee:
#     """Write to multiple file-like objects (e.g., console + file)."""
#     def __init__(self, *streams):
#         self.streams = streams
#     def write(self, data):
#         for s in self.streams:
#             s.write(data)
#             s.flush()
#     def flush(self):
#         for s in self.streams:
#             s.flush()

# def setup_run_logging(out_dir: str):
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     log_path = out_dir / "train.log"
#     # logging: both console and file
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[
#             logging.FileHandler(log_path, mode="a", encoding="utf-8"),
#             logging.StreamHandler(sys.__stdout__),
#         ],
#     )

#     # Tee stdout/stderr so all prints go into the same file
#     f = open(log_path, "a", encoding="utf-8", buffering=1)
#     sys.stdout = Tee(sys.__stdout__, f)
#     sys.stderr = Tee(sys.__stderr__, f)

#     logging.info(f"Logging to: {log_path}")
#     return str(log_path)


# # 每张图 semantic 前景面积（你现在 crag 一图一个 semantic 的话，直接就是 area）
# fg = ann.groupby("slide_uid")["area"].sum().reindex(meta["slide_uid"]).fillna(0).to_numpy()

# print("num slides:", len(fg))
# print("empty gt ratio:", (fg==0).mean())
# print("area stats:", np.percentile(fg, [0,25,50,75,90,99]))


# os.environ["HYDRA_FULL_ERROR"] = "1"


# def add_pythonpath_to_sys_path():
#     if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
#         return
#     sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


# def makedir(p: str):
#     Path(p).mkdir(parents=True, exist_ok=True)


# def format_exception(e: Exception, limit=30):
#     tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
#     return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


# def single_proc_run(local_rank: int, main_port: int, cfg, world_size: int):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(main_port)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     os.environ["WORLD_SIZE"] = str(world_size)

#     trainer = instantiate(cfg.trainer, _recursive_=False)
#     trainer.run()


# def single_node_runner(cfg, main_port: int):
#     assert cfg.launcher.num_nodes == 1
#     num_proc = int(cfg.launcher.gpus_per_node)

#     torch.multiprocessing.set_start_method("spawn", force=True)

#     if num_proc == 1:
#         single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=1)
#     else:
#         torch.multiprocessing.start_processes(
#             single_proc_run,
#             args=(main_port, cfg, num_proc),
#             nprocs=num_proc,
#             start_method="spawn",
#         )


# class SubmititRunner(submitit.helpers.Checkpointable):
#     def __init__(self, port: int, cfg):
#         self.cfg = cfg
#         self.port = port

#     def __call__(self):
#         job_env = submitit.JobEnvironment()
#         add_pythonpath_to_sys_path()

#         os.environ["MASTER_ADDR"] = job_env.hostnames[0]
#         os.environ["MASTER_PORT"] = str(self.port)
#         os.environ["RANK"] = str(job_env.global_rank)
#         os.environ["LOCAL_RANK"] = str(job_env.local_rank)
#         os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

#         try:
#             trainer = instantiate(self.cfg.trainer, _recursive_=False)
#             trainer.run()
#         except Exception as e:
#             logging.error(format_exception(e))
#             raise


# def main(args):
#     cfg = compose(config_name=args.config)

#     # cmdline override（和 SAM3 一样：优先命令行）
#     if args.num_gpus is not None:
#         cfg.launcher.gpus_per_node = int(args.num_gpus)
#     if args.num_nodes is not None:
#         cfg.launcher.num_nodes = int(args.num_nodes)
#     if args.use_cluster is not None:
#         cfg.submitit.use_cluster = bool(args.use_cluster)

#     # log dir
#     exp_dir = cfg.launcher.experiment_log_dir
#     if exp_dir is None:
#         exp_dir = str(Path(os.getcwd()) / "runs" / args.config)
#         cfg.launcher.experiment_log_dir = exp_dir
#     makedir(exp_dir)

#     # dump configs
#     print("###################### Train App Config ####################")
#     print(OmegaConf.to_yaml(cfg))
#     print("############################################################")

#     with open(Path(exp_dir) / "config.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg))

#     cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
#     with open(Path(exp_dir) / "config_resolved.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

#     add_pythonpath_to_sys_path()

#     # launch
#     if bool(cfg.submitit.use_cluster):
#         submitit_dir = str(Path(exp_dir) / "submitit_logs")
#         executor = submitit.AutoExecutor(folder=submitit_dir)

#         # 基本 job 参数（你按集群规范改 partition/account/qos）
#         job_kwargs = {
#             "timeout_min": int(cfg.submitit.timeout_hour) * 60,
#             "name": cfg.submitit.name if "name" in cfg.submitit else args.config,
#             "slurm_partition": cfg.submitit.partition,
#             "gpus_per_node": int(cfg.launcher.gpus_per_node),
#             "tasks_per_node": int(cfg.launcher.gpus_per_node),
#             "cpus_per_task": int(cfg.submitit.cpus_per_task),
#             "nodes": int(cfg.launcher.num_nodes),
#         }
#         executor.update_parameters(**job_kwargs)

#         port = random.randint(int(cfg.submitit.port_range[0]), int(cfg.submitit.port_range[1]))
#         runner = SubmititRunner(port, cfg)
#         job = executor.submit(runner)
#         print("Submitit Job ID:", job.job_id)
#     else:
#         cfg.launcher.num_nodes = 1
#         port = random.randint(int(cfg.submitit.port_range[0]), int(cfg.submitit.port_range[1]))
#         single_node_runner(cfg, port)


# if __name__ == "__main__":
#     # 让 hydra 用 module 方式找 configs（类似 sam3.train）
#     initialize_config_module("dataflow_train.configs", version_base="1.2")

#     parser = ArgumentParser()
#     parser.add_argument("-c", "--config", required=True, type=str, help="config name under dataflow_train/configs")
#     parser.add_argument("--use-cluster", type=int, default=None, help="0 local / 1 slurm")
#     parser.add_argument("--num-gpus", type=int, default=None)
#     parser.add_argument("--num-nodes", type=int, default=None)
#     args = parser.parse_args()
#     args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

#     main(args)



# dataflow_train/train.py
# A self-contained Hydra entry + semantic UNet trainer for parquet_db/v1
# Features:
#   1) Tee stdout/stderr to log file
#   2) Save side-by-side visualizations (RGB | GT | Pred) on FULL image (tiled inference)
#   3) Correct ROI paste-back for rle_roi=bbox masks (paste to full canvas then crop)

from __future__ import annotations

import os
import sys
import math
import time
import json
import random
import logging
import traceback
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import submitit
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf


# =========================
# Logging (stdout/stderr tee)
# =========================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_run_logging(out_dir: str, rank: int = 0) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / ("train.log" if rank == 0 else f"train_rank{rank}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.__stdout__),
        ],
    )

    f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)

    logging.info(f"[rank{rank}] Logging to: {log_path}")
    return str(log_path)


def format_exception(e: Exception, limit=40):
    tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def makedir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


# =========================
# RLE decode (COCO style counts, Fortran order)
# =========================
def rle_decode_counts(counts: List[int], h: int, w: int) -> np.ndarray:
    counts = [int(x) for x in counts]
    total = h * w
    if sum(counts) != total:
        raise ValueError(f"sum(counts)={sum(counts)} != h*w={total}")

    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        if val == 1 and run > 0:
            flat[idx : idx + run] = 1
        idx += run
        val ^= 1

    return flat.reshape((h, w), order="F").astype(bool)


# =========================
# Visualization utils
# =========================
def tensor_to_uint8_img(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    if x.max() <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def overlay_mask(img_uint8: np.ndarray, mask_bool: np.ndarray, color=(255, 0, 0), alpha=0.55) -> np.ndarray:
    img = img_uint8.astype(np.float32).copy()
    m = mask_bool.astype(bool)
    if m.ndim == 2:
        m = m[..., None]
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    img[m[..., 0]] = img[m[..., 0]] * (1 - alpha) + color_arr * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def save_side_by_side_full(
    img_u8: np.ndarray,
    gt_full: np.ndarray,
    pred_full_prob: np.ndarray,
    out_path: str,
    thr: float = 0.5,
    max_side: int = 1600,
):
    gt = gt_full.astype(bool)
    pred = (pred_full_prob >= thr)

    gt_vis = overlay_mask(img_u8, gt, color=(0, 255, 0), alpha=0.55)   # GT green
    pr_vis = overlay_mask(img_u8, pred, color=(255, 0, 0), alpha=0.55)  # Pred red

    gt_bw = (gt.astype(np.uint8) * 255)
    pr_bw = (pred.astype(np.uint8) * 255)
    gt_bw = np.stack([gt_bw] * 3, axis=-1)
    pr_bw = np.stack([pr_bw] * 3, axis=-1)

    panels = [img_u8, gt_vis, pr_vis, gt_bw, pr_bw]
    panel_imgs = [Image.fromarray(p) for p in panels]

    # optional downscale to keep files manageable
    def downscale(im: Image.Image) -> Image.Image:
        w, h = im.size
        s = max(w, h)
        if s <= max_side:
            return im
        scale = max_side / float(s)
        nw, nh = int(w * scale), int(h * scale)
        return im.resize((nw, nh), Image.BILINEAR)

    panel_imgs = [downscale(im) for im in panel_imgs]

    W = sum(im.size[0] for im in panel_imgs)
    H = max(im.size[1] for im in panel_imgs)
    canvas = Image.new("RGB", (W, H))
    x = 0
    for im in panel_imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# =========================
# Minimal UNet (good enough to validate pipeline)
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


# =========================
# Loss / metrics
# =========================
def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # logits: [B,1,H,W], target: [B,1,H,W] float {0,1}
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return dice.mean()


def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()


# =========================
# Parquet DB loader + Dataset with correct ROI paste-back
# =========================
def _safe_relpath_join(root: Path, rel_path: str) -> Path:
    return (root / rel_path).resolve()


class ParquetSemanticCropDataset(torch.utils.data.Dataset):
    """
    Training dataset that returns 512 crops (or patch_size) with GT mask aligned correctly.

    Key fix for rle_roi=bbox:
      - Each annotation stores ROI coords + RLE counts over that ROI
      - For a crop, we "paste ROI mask back to full canvas" conceptually by
        computing intersection ROI∩crop and placing the intersected region into crop mask.
      - This is equivalent to: (full_mask <- paste all ROI masks) then crop.
    """

    def __init__(
        self,
        db_root: str,
        dataset_roots: Dict[str, str],
        datasets: List[str],
        split: str,
        ann_file: str = "ann_semantic.parquet",
        patch_size: int = 512,
        epoch_size: int = 2000,
        max_slides: int = 0,
        pos_fraction: float = 0.7,
        seed: int = 42,
        cache_slides: int = 64,
    ):
        super().__init__()
        self.db_root = Path(db_root)
        self.dataset_roots = {k: Path(v) for k, v in dataset_roots.items()}
        self.datasets = [d.lower() for d in datasets]
        self.split = split
        self.ann_file = ann_file
        self.patch_size = int(patch_size)
        self.epoch_size = int(epoch_size)
        self.pos_fraction = float(pos_fraction)
        self.rng = random.Random(seed)
        self.cache_slides = int(cache_slides)

        meta_list = []
        ann_list = []

        for ds in self.datasets:
            meta_path = self.db_root / ds / "meta.parquet"
            ann_path = self.db_root / ds / ann_file
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing: {meta_path}")
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing: {ann_path}")

            meta = pd.read_parquet(
                meta_path,
                columns=["slide_uid", "dataset", "rel_path", "width_px", "height_px", "split"],
            )
            ann = pd.read_parquet(
                ann_path,
                columns=["ann_uid", "slide_uid", "label_id", "label_name", "roi_x", "roi_y", "roi_w", "roi_h",
                         "rle_size_h", "rle_size_w", "rle_counts", "area"],
            )
            meta_list.append(meta)
            ann_list.append(ann)

        meta_all = pd.concat(meta_list, ignore_index=True)
        ann_all = pd.concat(ann_list, ignore_index=True)

        def norm_split(s: str) -> str:
            s = str(s).lower()
            if s in ("val", "valid", "validation"):
                return "val"
            if s in ("train", "tr"):
                return "train"
            if s in ("test", "te"):
                return "test"
            return s

        meta_all["split"] = meta_all["split"].map(norm_split)
        split_norm = norm_split(split)

        self.meta = meta_all[meta_all["split"] == split_norm].reset_index(drop=True)
        if max_slides and max_slides > 0:
            self.meta = self.meta.head(int(max_slides)).reset_index(drop=True)

        # group annotations per slide_uid
        self.ann_by_slide: Dict[str, List[Dict[str, Any]]] = {}
        for r in ann_all.to_dict("records"):
            suid = str(r["slide_uid"])
            self.ann_by_slide.setdefault(suid, []).append(r)

        self.slide_uids: List[str] = self.meta["slide_uid"].astype(str).tolist()

        # slide-level cache: slide_uid -> list of decoded roi masks (roi_x,roi_y,mask_bool)
        self._slide_cache: "OrderedDict[str, List[Tuple[int,int,np.ndarray,int,int]]]" = OrderedDict()

        logging.info(f"[{split_norm}] slides={len(self.slide_uids)} ann_file={ann_file} patch={self.patch_size}")

    def __len__(self):
        # We define an "epoch" length in terms of number of sampled patches
        return self.epoch_size if self.split == "train" else len(self.slide_uids)

    def _load_image_full(self, meta_row: Dict[str, Any]) -> np.ndarray:
        ds = str(meta_row["dataset"]).lower()
        root = self.dataset_roots.get(ds, None)
        if root is None:
            raise KeyError(f"dataset_roots missing key: {ds}")
        p = _safe_relpath_join(root, str(meta_row["rel_path"]))
        img = Image.open(p).convert("RGB")
        return np.array(img)

    def _get_slide_decoded_rois(self, slide_uid: str) -> List[Tuple[int, int, np.ndarray, int, int]]:
        """
        Returns list of (roi_x, roi_y, roi_mask_bool, roi_w, roi_h).
        Cached per slide.
        """
        if slide_uid in self._slide_cache:
            self._slide_cache.move_to_end(slide_uid)
            return self._slide_cache[slide_uid]

        anns = self.ann_by_slide.get(slide_uid, [])
        decoded = []
        for a in anns:
            rx, ry = int(a["roi_x"]), int(a["roi_y"])
            rw, rh = int(a["roi_w"]), int(a["roi_h"])
            h, w = int(a["rle_size_h"]), int(a["rle_size_w"])
            counts = a["rle_counts"]
            if counts is None:
                continue
            # decode ROI mask
            m = rle_decode_counts(counts, h, w)
            # sanity: for rle_roi=bbox, rle_size should match roi_h/roi_w
            # but we won't hard-fail; we just use rle_size for mask itself.
            decoded.append((rx, ry, m, w, h))

        self._slide_cache[slide_uid] = decoded
        self._slide_cache.move_to_end(slide_uid)
        while len(self._slide_cache) > self.cache_slides:
            self._slide_cache.popitem(last=False)
        return decoded

    def _make_crop_mask(self, slide_uid: str, cx: int, cy: int, ps: int, H: int, W: int) -> np.ndarray:
        """
        Build crop mask [ps,ps] by pasting intersected ROI parts into crop.
        This is equivalent to (paste ROI to full canvas) then crop.
        """
        crop_mask = np.zeros((ps, ps), dtype=bool)
        decoded_rois = self._get_slide_decoded_rois(slide_uid)

        x0, y0 = int(cx), int(cy)
        x1, y1 = x0 + ps, y0 + ps

        for (rx, ry, roi_mask, mw, mh) in decoded_rois:
            # ROI box in full image coordinates: [rx, rx+mw) x [ry, ry+mh)
            ax0, ay0 = rx, ry
            ax1, ay1 = rx + mw, ry + mh

            ix0 = max(x0, ax0)
            iy0 = max(y0, ay0)
            ix1 = min(x1, ax1)
            iy1 = min(y1, ay1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            # slice within ROI mask coords
            roi_sx0 = ix0 - ax0
            roi_sy0 = iy0 - ay0
            roi_sx1 = ix1 - ax0
            roi_sy1 = iy1 - ay0

            sub = roi_mask[roi_sy0:roi_sy1, roi_sx0:roi_sx1]
            if sub.size == 0:
                continue

            # paste into crop coords
            cx0 = ix0 - x0
            cy0 = iy0 - y0
            cx1 = cx0 + sub.shape[1]
            cy1 = cy0 + sub.shape[0]
            crop_mask[cy0:cy1, cx0:cx1] |= sub

        return crop_mask

    def _sample_crop_xy(self, slide_uid: str, H: int, W: int) -> Tuple[int, int]:
        ps = self.patch_size
        if H <= ps or W <= ps:
            return 0, 0

        # Positive-biased sampling:
        # choose a crop that intersects at least one ROI bbox, without decoding full mask.
        if self.split == "train" and (self.rng.random() < self.pos_fraction):
            decoded_rois = self._get_slide_decoded_rois(slide_uid)
            if decoded_rois:
                rx, ry, m, mw, mh = self.rng.choice(decoded_rois)
                # choose crop top-left so that it overlaps ROI
                xmin = max(0, rx - ps + 1)
                xmax = min(W - ps, rx + mw - 1)
                ymin = max(0, ry - ps + 1)
                ymax = min(H - ps, ry + mh - 1)
                if xmin <= xmax and ymin <= ymax:
                    cx = self.rng.randint(xmin, xmax)
                    cy = self.rng.randint(ymin, ymax)
                    return cx, cy

        # fallback random crop
        cx = self.rng.randint(0, W - ps)
        cy = self.rng.randint(0, H - ps)
        return cx, cy

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == "train":
            slide_uid = self.slide_uids[idx % len(self.slide_uids)]
        else:
            slide_uid = self.slide_uids[idx]

        # meta row lookup
        mr = self.meta[self.meta["slide_uid"].astype(str) == slide_uid].iloc[0].to_dict()
        img_full = self._load_image_full(mr)  # [H,W,3] uint8
        H, W = img_full.shape[0], img_full.shape[1]
        ps = self.patch_size

        cx, cy = self._sample_crop_xy(slide_uid, H, W)
        crop = img_full[cy:cy + ps, cx:cx + ps]
        if crop.shape[0] != ps or crop.shape[1] != ps:
            # pad if needed (rare)
            pad = np.zeros((ps, ps, 3), dtype=np.uint8)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad

        crop_mask = self._make_crop_mask(slide_uid, cx, cy, ps, H=H, W=W).astype(np.uint8)

        # to tensor
        img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0  # [3,ps,ps]
        m_t = torch.from_numpy(crop_mask)[None, ...].float()             # [1,ps,ps]

        return {
            "image": img_t,
            "mask": m_t,
            "slide_uid": slide_uid,
            "crop_xy": (cx, cy),
            "full_hw": (H, W),
            "dataset": str(mr["dataset"]),
        }


# =========================
# Full-image GT builder + tiled inference
# =========================
def build_full_gt_mask(dataset: ParquetSemanticCropDataset, slide_uid: str, H: int, W: int) -> np.ndarray:
    """
    Assemble full GT mask by pasting decoded ROI masks into full canvas.
    Used only for debug visualization.
    """
    full = np.zeros((H, W), dtype=bool)
    decoded_rois = dataset._get_slide_decoded_rois(slide_uid)
    for (rx, ry, roi_mask, mw, mh) in decoded_rois:
        x0, y0 = rx, ry
        x1, y1 = min(W, rx + roi_mask.shape[1]), min(H, ry + roi_mask.shape[0])
        sub = roi_mask[: y1 - y0, : x1 - x0]
        full[y0:y1, x0:x1] |= sub
    return full


@torch.no_grad()
def tiled_predict_full(
    model: nn.Module,
    img_u8: np.ndarray,
    device: torch.device,
    patch: int = 512,
    stride: int = 256,
    amp: bool = True,
) -> np.ndarray:
    """
    Predict probability map on full image using overlapping tiles.
    Returns pred_prob [H,W] float32 in [0,1].
    """
    H, W = img_u8.shape[0], img_u8.shape[1]
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)

    model.eval()

    def iter_tiles():
        ys = list(range(0, max(1, H - patch + 1), stride))
        xs = list(range(0, max(1, W - patch + 1), stride))
        if ys[-1] != H - patch:
            ys.append(max(0, H - patch))
        if xs[-1] != W - patch:
            xs.append(max(0, W - patch))
        for y in ys:
            for x in xs:
                yield x, y

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda")

    for x, y in iter_tiles():
        tile = img_u8[y:y + patch, x:x + patch]
        if tile.shape[0] != patch or tile.shape[1] != patch:
            pad = np.zeros((patch, patch, 3), dtype=np.uint8)
            pad[: tile.shape[0], : tile.shape[1]] = tile
            tile = pad

        t = torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(t)  # [1,1,patch,patch]
            prob = torch.sigmoid(logits)[0, 0].float().detach().cpu().numpy()

        yh = min(H, y + patch)
        xw = min(W, x + patch)
        ph = yh - y
        pw = xw - x
        pred_sum[y:yh, x:xw] += prob[:ph, :pw]
        pred_cnt[y:yh, x:xw] += 1.0

    pred = pred_sum / np.maximum(pred_cnt, 1e-6)
    return pred


# =========================
# Trainer
# =========================
class SemanticUNetTrainer:
    """
    A trainer that can be instantiated by Hydra:
      trainer:
        _target_: dataflow_train.train.SemanticUNetTrainer
        ...
    """

    def __init__(
        self,
        out_dir: str,
        db_root: str,
        dataset_roots: Dict[str, str],
        datasets: List[str],
        ann_file: str = "ann_semantic.parquet",
        patch_size: int = 512,
        epochs: int = 5,
        train_epoch_size: int = 2000,
        batch_size: int = 8,
        num_workers: int = 4,
        lr: float = 3e-4,
        wd: float = 1e-4,
        seed: int = 42,
        amp: bool = True,
        log_freq: int = 20,
        debug_vis_full: int = 4,
        debug_vis_thr: float = 0.5,
        debug_vis_max_side: int = 1600,
        stride: int = 256,
        pos_fraction: float = 0.7,
        cache_slides: int = 64,
    ):
        self.out_dir = str(out_dir)
        self.db_root = str(db_root)
        self.dataset_roots = dataset_roots
        self.datasets = datasets
        self.ann_file = ann_file
        self.patch_size = int(patch_size)
        self.epochs = int(epochs)
        self.train_epoch_size = int(train_epoch_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.lr = float(lr)
        self.wd = float(wd)
        self.seed = int(seed)
        self.amp = bool(amp)
        self.log_freq = int(log_freq)

        self.debug_vis_full = int(debug_vis_full)
        self.debug_vis_thr = float(debug_vis_thr)
        self.debug_vis_max_side = int(debug_vis_max_side)
        self.stride = int(stride)

        self.pos_fraction = float(pos_fraction)
        self.cache_slides = int(cache_slides)

        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # seed
        random.seed(self.seed + self.rank)
        np.random.seed(self.seed + self.rank)
        torch.manual_seed(self.seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + self.rank)

    def _is_main(self) -> bool:
        return self.rank == 0

    def _init_dist(self):
        if self.world_size <= 1:
            return
        if torch.distributed.is_initialized():
            return
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(self.local_rank)

    def _build_loaders(self):
        train_ds = ParquetSemanticCropDataset(
            db_root=self.db_root,
            dataset_roots=self.dataset_roots,
            datasets=self.datasets,
            split="train",
            ann_file=self.ann_file,
            patch_size=self.patch_size,
            epoch_size=self.train_epoch_size,
            pos_fraction=self.pos_fraction,
            seed=self.seed + 123,
            cache_slides=self.cache_slides,
        )
        val_ds = ParquetSemanticCropDataset(
            db_root=self.db_root,
            dataset_roots=self.dataset_roots,
            datasets=self.datasets,
            split="val",
            ann_file=self.ann_file,
            patch_size=self.patch_size,
            epoch_size=0,
            pos_fraction=0.0,
            seed=self.seed + 999,
            cache_slides=self.cache_slides,
        )

        if self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )
        return train_ds, val_ds, train_loader, val_loader

    def _build_model(self):
        model = UNetSmall(in_ch=3, out_ch=1, base=32)
        return model

    def _save_ckpt(self, model, optim, epoch, best_score, name: str):
        if not self._is_main():
            return
        ckpt = {
            "epoch": epoch,
            "best_score": float(best_score),
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        }
        p = Path(self.out_dir) / "checkpoints"
        p.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(p / name))

    def _debug_full_vis(self, model, val_ds: ParquetSemanticCropDataset, epoch: int, device: torch.device):
        if not self._is_main():
            return
        if self.debug_vis_full <= 0:
            return

        out_dir = Path(self.out_dir) / "debug_vis_full" / f"epoch{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # pick a few val slides
        slide_uids = val_ds.slide_uids
        if not slide_uids:
            logging.info("[debug_vis] no val slides")
            return

        pick = slide_uids[: min(self.debug_vis_full, len(slide_uids))]

        for suid in pick:
            # load full image
            mr = val_ds.meta[val_ds.meta["slide_uid"].astype(str) == suid].iloc[0].to_dict()
            img_full = val_ds._load_image_full(mr)
            H, W = img_full.shape[0], img_full.shape[1]

            # full GT mask (real paste-back)
            gt_full = build_full_gt_mask(val_ds, suid, H=H, W=W)

            # full pred by tiled inference
            pred_full = tiled_predict_full(
                model=model,
                img_u8=img_full,
                device=device,
                patch=self.patch_size,
                stride=self.stride,
                amp=self.amp,
            )

            safe_name = suid.replace("/", "_").replace(":", "_")
            out_path = str(out_dir / f"{safe_name}.png")
            save_side_by_side_full(
                img_u8=img_full,
                gt_full=gt_full,
                pred_full_prob=pred_full,
                out_path=out_path,
                thr=self.debug_vis_thr,
                max_side=self.debug_vis_max_side,
            )

        logging.info(f"[debug_vis] saved full side-by-side to: {out_dir}")

    def run(self):
        self._init_dist()

        device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(self.local_rank)

        # logging (per-rank safe)
        setup_run_logging(self.out_dir, rank=self.rank)
        logging.info(f"device={device} rank={self.rank} world={self.world_size}")

        train_ds, val_ds, train_loader, val_loader = self._build_loaders()

        model = self._build_model().to(device)
        if self.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)

        optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        bce = nn.BCEWithLogitsLoss()

        best = -1.0

        scaler = torch.cuda.amp.GradScaler(enabled=(self.amp and device.type == "cuda"))

        for epoch in range(1, self.epochs + 1):
            if self.world_size > 1:
                # reshuffle each epoch
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

            # ---- train ----
            model.train()
            t0 = time.time()
            losses = []
            dices = []

            for it, batch in enumerate(train_loader, start=1):
                img = batch["image"].to(device, non_blocking=True)  # [B,3,ps,ps]
                msk = batch["mask"].to(device, non_blocking=True)   # [B,1,ps,ps]

                optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
                    logits = model(img)
                    loss = 0.7 * bce(logits, msk) + 0.3 * soft_dice_loss_from_logits(logits, msk)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                with torch.no_grad():
                    d = dice_score_from_logits(logits, msk).item()

                losses.append(loss.item())
                dices.append(d)

                if self._is_main() and (it % self.log_freq == 0):
                    logging.info(
                        f"Epoch {epoch:03d} | it {it:04d}/{len(train_loader)} "
                        f"loss {np.mean(losses):.4f} dice {np.mean(dices):.4f}"
                    )

            train_loss = float(np.mean(losses)) if losses else 0.0
            train_dice = float(np.mean(dices)) if dices else 0.0

            # ---- val ----
            model.eval()
            v_losses = []
            v_dices = []

            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(device, non_blocking=True)
                    msk = batch["mask"].to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
                        logits = model(img)
                        loss = 0.7 * bce(logits, msk) + 0.3 * soft_dice_loss_from_logits(logits, msk)

                    d = dice_score_from_logits(logits, msk).item()
                    v_losses.append(loss.item())
                    v_dices.append(d)

            val_loss = float(np.mean(v_losses)) if v_losses else 0.0
            val_dice = float(np.mean(v_dices)) if v_dices else 0.0

            # reduce across ranks (optional)
            if self.world_size > 1:
                tl = torch.tensor([train_loss, train_dice, val_loss, val_dice], device=device)
                torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
                tl = tl / float(self.world_size)
                train_loss, train_dice, val_loss, val_dice = [float(x) for x in tl.tolist()]

            if self._is_main():
                dt = time.time() - t0
                logging.info(
                    f"Epoch {epoch:03d} | train loss {train_loss:.4f} dice {train_dice:.4f} "
                    f"| val loss {val_loss:.4f} dice {val_dice:.4f} | {dt:.1f}s"
                )

            # checkpoints
            if val_dice > best:
                best = val_dice
                self._save_ckpt(model.module if hasattr(model, "module") else model, optim, epoch, best, "best.pt")
            self._save_ckpt(model.module if hasattr(model, "module") else model, optim, epoch, best, "last.pt")

            # full-image debug vis (GT vs Pred on original image)
            self._debug_full_vis(model.module if hasattr(model, "module") else model, val_ds, epoch, device)

        if self._is_main():
            logging.info(f"[DONE] best_dice={best:.4f} out_dir={self.out_dir}")


# =========================
# Launchers (local spawn + submitit)
# =========================
os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank: int, main_port: int, cfg, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    add_pythonpath_to_sys_path()

    # ensure trainer.out_dir exists and logging works per rank
    exp_dir = cfg.launcher.experiment_log_dir
    if exp_dir is None:
        exp_dir = str(Path(os.getcwd()) / "runs" / "unnamed")
        cfg.launcher.experiment_log_dir = exp_dir

    # inject out_dir into trainer if not present
    if "out_dir" not in cfg.trainer:
        cfg.trainer.out_dir = exp_dir

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert int(cfg.launcher.num_nodes) == 1
    num_proc = int(cfg.launcher.gpus_per_node)

    torch.multiprocessing.set_start_method("spawn", force=True)

    if num_proc == 1:
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=1)
    else:
        torch.multiprocessing.start_processes(
            single_proc_run,
            args=(main_port, cfg, num_proc),
            nprocs=num_proc,
            start_method="spawn",
        )


class SubmititRunner(submitit.helpers.Checkpointable):
    def __init__(self, port: int, cfg):
        self.cfg = cfg
        self.port = port

    def __call__(self):
        job_env = submitit.JobEnvironment()
        add_pythonpath_to_sys_path()

        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        try:
            # set out_dir for trainer
            exp_dir = self.cfg.launcher.experiment_log_dir
            if "out_dir" not in self.cfg.trainer:
                self.cfg.trainer.out_dir = exp_dir

            trainer = instantiate(self.cfg.trainer, _recursive_=False)
            trainer.run()
        except Exception as e:
            logging.error(format_exception(e))
            raise


def main(args):
    cfg = compose(config_name=args.config)

    # cmdline overrides
    if args.num_gpus is not None:
        cfg.launcher.gpus_per_node = int(args.num_gpus)
    if args.num_nodes is not None:
        cfg.launcher.num_nodes = int(args.num_nodes)
    if args.use_cluster is not None:
        cfg.submitit.use_cluster = bool(args.use_cluster)

    # exp dir
    exp_dir = cfg.launcher.experiment_log_dir
    if exp_dir is None:
        exp_dir = str(Path(os.getcwd()) / "runs" / args.config)
        cfg.launcher.experiment_log_dir = exp_dir
    makedir(exp_dir)

    # dump configs
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    with open(Path(exp_dir) / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open(Path(exp_dir) / "config_resolved.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    add_pythonpath_to_sys_path()

    # launch
    if bool(cfg.submitit.use_cluster):
        submitit_dir = str(Path(exp_dir) / "submitit_logs")
        executor = submitit.AutoExecutor(folder=submitit_dir)

        job_kwargs = {
            "timeout_min": int(cfg.submitit.timeout_hour) * 60,
            "name": cfg.submitit.name if "name" in cfg.submitit else args.config,
            "slurm_partition": cfg.submitit.partition,
            "gpus_per_node": int(cfg.launcher.gpus_per_node),
            "tasks_per_node": int(cfg.launcher.gpus_per_node),
            "cpus_per_task": int(cfg.submitit.cpus_per_task),
            "nodes": int(cfg.launcher.num_nodes),
        }
        executor.update_parameters(**job_kwargs)

        port = random.randint(int(cfg.submitit.port_range[0]), int(cfg.submitit.port_range[1]))
        runner = SubmititRunner(port, cfg)
        job = executor.submit(runner)
        print("Submitit Job ID:", job.job_id)
    else:
        cfg.launcher.num_nodes = 1
        port = random.randint(int(cfg.submitit.port_range[0]), int(cfg.submitit.port_range[1]))
        single_node_runner(cfg, port)


if __name__ == "__main__":
    # IMPORTANT:
    #   configs should live at: dataflow_train/configs/<name>.yaml
    #   and dataflow_train/configs/__init__.py should exist
    initialize_config_module("dataflow_train.configs", version_base="1.2")

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="config name under dataflow_train/configs")
    parser.add_argument("--use-cluster", type=int, default=None, help="0 local / 1 slurm")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    main(args)
