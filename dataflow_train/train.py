










# from __future__ import annotations

# import os
# import sys
# import time
# import json
# import random
# import logging
# import traceback
# from argparse import ArgumentParser
# from pathlib import Path
# from collections import OrderedDict
# from typing import Dict, List, Tuple, Optional, Any, Union

# import numpy as np
# import pandas as pd
# from PIL import Image

# import torch
# import torch.nn as nn

# import submitit
# from hydra import compose, initialize_config_module
# from hydra.utils import instantiate
# from omegaconf import OmegaConf


# # =========================
# # Logging (stdout/stderr tee)
# # =========================
# class Tee:
#     def __init__(self, *streams):
#         self.streams = streams

#     def write(self, data):
#         for s in self.streams:
#             s.write(data)
#             s.flush()

#     def flush(self):
#         for s in self.streams:
#             s.flush()


# def setup_run_logging(out_dir: str, rank: int = 0) -> str:
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     log_path = out_dir / ("train.log" if rank == 0 else f"train_rank{rank}.log")

#     root = logging.getLogger()
#     root.setLevel(logging.INFO)
#     for h in list(root.handlers):
#         root.removeHandler(h)

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[
#             logging.FileHandler(log_path, mode="a", encoding="utf-8"),
#             logging.StreamHandler(sys.__stdout__),
#         ],
#     )

#     f = open(log_path, "a", encoding="utf-8", buffering=1)
#     sys.stdout = Tee(sys.__stdout__, f)
#     sys.stderr = Tee(sys.__stderr__, f)

#     logging.info(f"[rank{rank}] Logging to: {log_path}")
#     return str(log_path)


# def format_exception(e: Exception, limit=80):
#     tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
#     return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


# def add_pythonpath_to_sys_path():
#     if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
#         return
#     sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


# def makedir(p: str):
#     Path(p).mkdir(parents=True, exist_ok=True)


# def now_ts() -> str:
#     return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


# # =========================
# # RLE decode (COCO style counts, Fortran order) - tolerant
# # =========================
# def _ensure_counts(counts: Union[str, List[int], Tuple[int, ...], np.ndarray]) -> List[int]:
#     if counts is None:
#         return []
#     if isinstance(counts, np.ndarray):
#         counts = counts.tolist()
#     if isinstance(counts, str):
#         counts = json.loads(counts)
#     return [int(x) for x in counts]


# def rle_decode_counts_tolerant(counts: Union[str, List[int], Tuple[int, ...], np.ndarray], h: int, w: int) -> np.ndarray:
#     """
#     COCO-style RLE counts, Fortran order flatten.
#     Tolerant: does NOT require sum(counts)==h*w. Clips if overshoot.
#     Returns bool mask [h,w].
#     """
#     counts = _ensure_counts(counts)
#     n = int(h) * int(w)
#     flat = np.zeros(n, dtype=np.uint8)

#     idx = 0
#     val = 0
#     for run in counts:
#         if run <= 0:
#             continue
#         end = idx + run
#         if end > n:
#             end = n
#         if val == 1:
#             flat[idx:end] = 1
#         idx = end
#         val ^= 1
#         if idx >= n:
#             break

#     return flat.reshape((h, w), order="F").astype(bool)


# # =========================
# # Visualization utils (crop + full)
# # =========================
# def overlay_mask(img_uint8: np.ndarray, mask_bool: np.ndarray, color=(255, 0, 0), alpha=0.55) -> np.ndarray:
#     img = img_uint8.astype(np.float32).copy()
#     m = mask_bool.astype(bool)
#     if m.ndim == 2:
#         m = m[..., None]
#     color_arr = np.array(color, dtype=np.float32)[None, None, :]
#     img[m[..., 0]] = img[m[..., 0]] * (1 - alpha) + color_arr * alpha
#     return np.clip(img, 0, 255).astype(np.uint8)


# def make_error_overlay(img_u8: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha=0.60) -> np.ndarray:
#     """
#     Error overlay:
#       TP: green, FP: red, FN: blue
#     """
#     gt = gt.astype(bool)
#     pred = pred.astype(bool)
#     tp = gt & pred
#     fp = (~gt) & pred
#     fn = gt & (~pred)

#     out = img_u8.astype(np.float32).copy()

#     def apply(mask, color):
#         if mask.ndim == 2:
#             mask_ = mask[..., None]
#         else:
#             mask_ = mask
#         c = np.array(color, dtype=np.float32)[None, None, :]
#         out[mask_[..., 0]] = out[mask_[..., 0]] * (1 - alpha) + c * alpha

#     apply(tp, (0, 255, 0))
#     apply(fp, (255, 0, 0))
#     apply(fn, (0, 0, 255))
#     return np.clip(out, 0, 255).astype(np.uint8)


# def _concat_panels_h(panels: List[np.ndarray], out_path: str, max_side: int = 1600):
#     ims = [Image.fromarray(p) for p in panels]

#     def downscale(im: Image.Image) -> Image.Image:
#         w, h = im.size
#         s = max(w, h)
#         if s <= max_side:
#             return im
#         scale = max_side / float(s)
#         nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
#         return im.resize((nw, nh), Image.BILINEAR)

#     ims = [downscale(im) for im in ims]
#     W = sum(im.size[0] for im in ims)
#     H = max(im.size[1] for im in ims)
#     canvas = Image.new("RGB", (W, H))
#     x = 0
#     for im in ims:
#         canvas.paste(im, (x, 0))
#         x += im.size[0]
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     canvas.save(out_path)


# def save_side_by_side_crop(img_u8: np.ndarray, gt: np.ndarray, pred_prob: np.ndarray, out_path: str, thr: float = 0.5, max_side: int = 1600):
#     gt = gt.astype(bool)
#     pred = (pred_prob >= thr)

#     gt_bw = np.stack([(gt.astype(np.uint8) * 255)] * 3, axis=-1)
#     pr_bw = np.stack([(pred.astype(np.uint8) * 255)] * 3, axis=-1)
#     err = make_error_overlay(img_u8, gt, pred, alpha=0.60)

#     panels = [img_u8, gt_bw, pr_bw, err]  # 原图 | GT | Pred | ErrorOverlay(TP/FP/FN)
#     _concat_panels_h(panels, out_path, max_side=max_side)


# def save_side_by_side_full(img_u8: np.ndarray, gt_full: np.ndarray, pred_full_prob: np.ndarray, out_path: str, thr: float = 0.5, max_side: int = 1600):
#     gt = gt_full.astype(bool)
#     pred = (pred_full_prob >= thr)

#     gt_vis = overlay_mask(img_u8, gt, color=(0, 255, 0), alpha=0.55)
#     pr_vis = overlay_mask(img_u8, pred, color=(255, 0, 0), alpha=0.55)
#     err = make_error_overlay(img_u8, gt, pred, alpha=0.55)

#     panels = [img_u8, gt_vis, pr_vis, err]  # 原图 | GT overlay | Pred overlay | ErrorOverlay
#     _concat_panels_h(panels, out_path, max_side=max_side)


# # =========================
# # Minimal UNet
# # =========================
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class UNetSmall(nn.Module):
#     def __init__(self, in_ch=3, out_ch=1, base=32):
#         super().__init__()
#         self.enc1 = ConvBlock(in_ch, base)
#         self.pool1 = nn.MaxPool2d(2)
#         self.enc2 = ConvBlock(base, base * 2)
#         self.pool2 = nn.MaxPool2d(2)
#         self.enc3 = ConvBlock(base * 2, base * 4)
#         self.pool3 = nn.MaxPool2d(2)

#         self.mid = ConvBlock(base * 4, base * 8)

#         self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
#         self.dec3 = ConvBlock(base * 8, base * 4)
#         self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
#         self.dec2 = ConvBlock(base * 4, base * 2)
#         self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
#         self.dec1 = ConvBlock(base * 2, base)

#         self.out = nn.Conv2d(base, out_ch, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))
#         m = self.mid(self.pool3(e3))

#         d3 = self.up3(m)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))
#         d2 = self.up2(d3)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))
#         d1 = self.up1(d2)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))
#         return self.out(d1)


# # =========================
# # Metrics helpers (avoid dice inflation)
# # =========================
# def dice_per_image_from_prob(prob: np.ndarray, gt: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> float:
#     gt = gt.astype(bool)
#     pred = (prob >= thr)
#     inter = float((pred & gt).sum())
#     den = float(pred.sum() + gt.sum())
#     return float((2.0 * inter + eps) / (den + eps))


# def summarize_array(x: List[float]) -> Dict[str, float]:
#     if len(x) == 0:
#         return {"mean": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
#     arr = np.array(x, dtype=np.float32)
#     return {
#         "mean": float(arr.mean()),
#         "p10": float(np.quantile(arr, 0.10)),
#         "p25": float(np.quantile(arr, 0.25)),
#         "p50": float(np.quantile(arr, 0.50)),
#         "p75": float(np.quantile(arr, 0.75)),
#         "p90": float(np.quantile(arr, 0.90)),
#     }


# # =========================
# # Parquet DB loader + Dataset (ROI∩crop -> crop mask)
# # =========================
# def _safe_relpath_join(root: Path, rel_path: str) -> Path:
#     return (root / rel_path).resolve()


# def _pick_first_existing_col(df: pd.DataFrame, candidates: List[str], must: bool = True) -> Optional[str]:
#     for c in candidates:
#         if c in df.columns:
#             return c
#     if must:
#         raise KeyError(f"Missing columns, tried: {candidates}. Existing head: {list(df.columns)[:50]}")
#     return None


# class ParquetSemanticCropDataset(torch.utils.data.Dataset):
#     """
#     Returns patch crops with correct GT alignment under rle_roi=bbox.
#     """

#     def __init__(
#         self,
#         db_root: str,
#         dataset_roots: Dict[str, str],
#         datasets: List[str],
#         split: str,
#         ann_file: str = "ann_semantic.parquet",
#         patch_size: int = 512,
#         epoch_size: int = 2000,
#         max_slides: int = 0,
#         pos_fraction: float = 0.7,
#         seed: int = 42,
#         cache_slides: int = 64,
#         use_meta_split: bool = True,
#         val_ratio: float = 0.1,
#     ):
#         super().__init__()
#         self.db_root = Path(db_root)
#         self.dataset_roots = {k.lower(): Path(v) for k, v in dataset_roots.items()}
#         self.datasets = [d.lower() for d in datasets]
#         self.split = str(split).lower()
#         self.ann_file = ann_file
#         self.patch_size = int(patch_size)
#         self.epoch_size = int(epoch_size)
#         self.pos_fraction = float(pos_fraction)
#         self.rng = random.Random(seed)
#         self.cache_slides = int(cache_slides)
#         self.use_meta_split = bool(use_meta_split)
#         self.val_ratio = float(val_ratio)

#         meta_list = []
#         ann_list = []

#         for ds in self.datasets:
#             meta_path = self.db_root / ds / "meta.parquet"
#             ann_path = self.db_root / ds / ann_file
#             if not meta_path.exists():
#                 raise FileNotFoundError(f"Missing: {meta_path}")
#             if not ann_path.exists():
#                 raise FileNotFoundError(f"Missing: {ann_path}")

#             meta = pd.read_parquet(meta_path)
#             ann = pd.read_parquet(ann_path)

#             uid_col = _pick_first_existing_col(meta, ["slide_uid", "uid", "image_uid"])
#             rel_col = _pick_first_existing_col(meta, ["rel_path", "path", "image_path"])
#             h_col = _pick_first_existing_col(meta, ["height_px", "height", "H", "h"])
#             w_col = _pick_first_existing_col(meta, ["width_px", "width", "W", "w"])

#             if "dataset" not in meta.columns:
#                 meta["dataset"] = ds

#             split_col = "split" if "split" in meta.columns else None

#             meta = meta.rename(columns={uid_col: "slide_uid", rel_col: "rel_path", h_col: "height_px", w_col: "width_px"})

#             ann_slide_col = _pick_first_existing_col(ann, ["slide_uid", "image_uid", "uid"])
#             rx_col = _pick_first_existing_col(ann, ["roi_x", "x", "bbox_x"])
#             ry_col = _pick_first_existing_col(ann, ["roi_y", "y", "bbox_y"])
#             sh_col = _pick_first_existing_col(ann, ["rle_size_h", "mask_h", "roi_h", "h"])
#             sw_col = _pick_first_existing_col(ann, ["rle_size_w", "mask_w", "roi_w", "w"])
#             cnt_col = _pick_first_existing_col(ann, ["rle_counts", "counts", "rle"])

#             ann = ann.rename(
#                 columns={
#                     ann_slide_col: "slide_uid",
#                     rx_col: "roi_x",
#                     ry_col: "roi_y",
#                     sh_col: "rle_size_h",
#                     sw_col: "rle_size_w",
#                     cnt_col: "rle_counts",
#                 }
#             )

#             # keep minimal cols
#             meta_keep = ["slide_uid", "dataset", "rel_path", "width_px", "height_px"] + (["split"] if split_col else [])
#             meta_list.append(meta[meta_keep])
#             ann_list.append(ann[["slide_uid", "roi_x", "roi_y", "rle_size_h", "rle_size_w", "rle_counts"]])

#         meta_all = pd.concat(meta_list, ignore_index=True)
#         ann_all = pd.concat(ann_list, ignore_index=True)

#         def norm_split(s: str) -> str:
#             s = str(s).lower()
#             if s in ("val", "valid", "validation"):
#                 return "val"
#             if s in ("train", "tr"):
#                 return "train"
#             if s in ("test", "te"):
#                 return "test"
#             return s

#         split_norm = norm_split(self.split)

#         if self.use_meta_split and "split" in meta_all.columns:
#             meta_all["split"] = meta_all["split"].map(norm_split)
#             meta_f = meta_all[meta_all["split"] == split_norm].reset_index(drop=True)
#         else:
#             uids = meta_all["slide_uid"].astype(str).unique().tolist()
#             uids.sort()
#             rng = np.random.default_rng(12345)
#             rng.shuffle(uids)
#             n_val = int(len(uids) * self.val_ratio)
#             val_set = set(uids[:n_val])
#             if split_norm == "val":
#                 keep = meta_all["slide_uid"].astype(str).isin(val_set)
#             elif split_norm == "train":
#                 keep = ~meta_all["slide_uid"].astype(str).isin(val_set)
#             else:
#                 keep = np.ones(len(meta_all), dtype=bool)
#             meta_f = meta_all[keep].reset_index(drop=True)

#         if max_slides and max_slides > 0:
#             meta_f = meta_f.head(int(max_slides)).reset_index(drop=True)

#         self.meta = meta_f

#         # group annotations per slide_uid
#         self.ann_by_slide: Dict[str, List[Dict[str, Any]]] = {}
#         for r in ann_all.to_dict("records"):
#             suid = str(r["slide_uid"])
#             self.ann_by_slide.setdefault(suid, []).append(r)

#         self.slide_uids: List[str] = self.meta["slide_uid"].astype(str).tolist()

#         # slide cache: slide_uid -> list of (roi_x, roi_y, roi_mask_bool, roi_w, roi_h)
#         self._slide_cache: "OrderedDict[str, List[Tuple[int,int,np.ndarray,int,int]]]" = OrderedDict()

#         logging.info(f"[{split_norm}] slides={len(self.slide_uids)} ann_file={ann_file} patch={self.patch_size}")

#     def __len__(self):
#         return self.epoch_size if self.split == "train" else len(self.slide_uids)

#     def _load_image_full(self, meta_row: Dict[str, Any]) -> np.ndarray:
#         ds = str(meta_row["dataset"]).lower()
#         root = self.dataset_roots.get(ds, None)
#         if root is None:
#             raise KeyError(f"dataset_roots missing key: {ds}. Provided={list(self.dataset_roots.keys())}")
#         p = _safe_relpath_join(root, str(meta_row["rel_path"]))
#         if not p.exists():
#             raise FileNotFoundError(f"Image not found: {p}")
#         return np.array(Image.open(p).convert("RGB"))

#     def _get_slide_decoded_rois(self, slide_uid: str) -> List[Tuple[int, int, np.ndarray, int, int]]:
#         if slide_uid in self._slide_cache:
#             self._slide_cache.move_to_end(slide_uid)
#             return self._slide_cache[slide_uid]

#         anns = self.ann_by_slide.get(slide_uid, [])
#         decoded: List[Tuple[int, int, np.ndarray, int, int]] = []
#         for a in anns:
#             rx, ry = int(a["roi_x"]), int(a["roi_y"])
#             h, w = int(a["rle_size_h"]), int(a["rle_size_w"])
#             counts = a["rle_counts"]
#             if counts is None:
#                 continue
#             m = rle_decode_counts_tolerant(counts, h, w)
#             decoded.append((rx, ry, m, w, h))

#         self._slide_cache[slide_uid] = decoded
#         self._slide_cache.move_to_end(slide_uid)
#         while len(self._slide_cache) > self.cache_slides:
#             self._slide_cache.popitem(last=False)
#         return decoded

#     def _make_crop_mask_fast(self, slide_uid: str, cx: int, cy: int, ps: int) -> np.ndarray:
#         """ROI∩crop -> crop mask (训练/验证用)"""
#         crop_mask = np.zeros((ps, ps), dtype=bool)
#         decoded_rois = self._get_slide_decoded_rois(slide_uid)

#         x0, y0 = int(cx), int(cy)
#         x1, y1 = x0 + ps, y0 + ps

#         for (rx, ry, roi_mask, mw, mh) in decoded_rois:
#             ax0, ay0 = rx, ry
#             ax1, ay1 = rx + mw, ry + mh

#             ix0 = max(x0, ax0)
#             iy0 = max(y0, ay0)
#             ix1 = min(x1, ax1)
#             iy1 = min(y1, ay1)
#             if ix1 <= ix0 or iy1 <= iy0:
#                 continue

#             roi_sx0 = ix0 - ax0
#             roi_sy0 = iy0 - ay0
#             roi_sx1 = ix1 - ax0
#             roi_sy1 = iy1 - ay0

#             sub = roi_mask[roi_sy0:roi_sy1, roi_sx0:roi_sx1]
#             if sub.size == 0:
#                 continue

#             cx0 = ix0 - x0
#             cy0 = iy0 - y0
#             cx1 = cx0 + sub.shape[1]
#             cy1 = cy0 + sub.shape[0]
#             crop_mask[cy0:cy1, cx0:cx1] |= sub

#         return crop_mask

#     def build_full_mask_slow(self, slide_uid: str, H: int, W: int) -> np.ndarray:
#         """full canvas paste (只用于 sanity/debug)"""
#         full = np.zeros((H, W), dtype=bool)
#         decoded_rois = self._get_slide_decoded_rois(slide_uid)
#         for (rx, ry, roi_mask, mw, mh) in decoded_rois:
#             x0, y0 = rx, ry
#             x1 = min(W, x0 + roi_mask.shape[1])
#             y1 = min(H, y0 + roi_mask.shape[0])
#             sub = roi_mask[: y1 - y0, : x1 - x0]
#             if sub.size:
#                 full[y0:y1, x0:x1] |= sub
#         return full

#     def _sample_crop_xy(self, slide_uid: str, H: int, W: int) -> Tuple[int, int]:
#         ps = self.patch_size
#         if H <= ps or W <= ps:
#             return 0, 0

#         if self.split == "train" and (self.rng.random() < self.pos_fraction):
#             decoded_rois = self._get_slide_decoded_rois(slide_uid)
#             if decoded_rois:
#                 rx, ry, m, mw, mh = self.rng.choice(decoded_rois)
#                 xmin = max(0, rx - ps + 1)
#                 xmax = min(W - ps, rx + mw - 1)
#                 ymin = max(0, ry - ps + 1)
#                 ymax = min(H - ps, ry + mh - 1)
#                 if xmin <= xmax and ymin <= ymax:
#                     return self.rng.randint(xmin, xmax), self.rng.randint(ymin, ymax)

#         return self.rng.randint(0, W - ps), self.rng.randint(0, H - ps)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         if self.split == "train":
#             slide_uid = self.slide_uids[idx % len(self.slide_uids)]
#         else:
#             slide_uid = self.slide_uids[idx]

#         mr = self.meta[self.meta["slide_uid"].astype(str) == slide_uid].iloc[0].to_dict()
#         img_full = self._load_image_full(mr)
#         H, W = img_full.shape[0], img_full.shape[1]
#         ps = self.patch_size

#         cx, cy = self._sample_crop_xy(slide_uid, H, W)
#         crop = img_full[cy:cy + ps, cx:cx + ps]
#         if crop.shape[0] != ps or crop.shape[1] != ps:
#             pad = np.zeros((ps, ps, 3), dtype=np.uint8)
#             pad[: crop.shape[0], : crop.shape[1]] = crop
#             crop = pad

#         crop_mask = self._make_crop_mask_fast(slide_uid, cx, cy, ps).astype(np.uint8)

#         img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
#         m_t = torch.from_numpy(crop_mask)[None, ...].float()

#         return {
#             "image": img_t,
#             "mask": m_t,
#             "slide_uid": slide_uid,
#             "crop_xy": (cx, cy),
#             "full_hw": (H, W),
#             "dataset": str(mr.get("dataset", "")),
#             "rel_path": str(mr.get("rel_path", "")),
#         }


# # =========================
# # Full-image tiled inference (for debug)
# # =========================
# @torch.no_grad()
# def tiled_predict_full(model: nn.Module, img_u8: np.ndarray, device: torch.device, patch: int = 512, stride: int = 256, amp: bool = True) -> np.ndarray:
#     H, W = img_u8.shape[0], img_u8.shape[1]
#     pred_sum = np.zeros((H, W), dtype=np.float32)
#     pred_cnt = np.zeros((H, W), dtype=np.float32)

#     model.eval()

#     def iter_tiles():
#         ys = list(range(0, max(1, H - patch + 1), stride))
#         xs = list(range(0, max(1, W - patch + 1), stride))
#         if ys and ys[-1] != H - patch:
#             ys.append(max(0, H - patch))
#         if xs and xs[-1] != W - patch:
#             xs.append(max(0, W - patch))
#         for y in ys:
#             for x in xs:
#                 yield x, y

#     autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda")

#     for x, y in iter_tiles():
#         tile = img_u8[y:y + patch, x:x + patch]
#         if tile.shape[0] != patch or tile.shape[1] != patch:
#             pad = np.zeros((patch, patch, 3), dtype=np.uint8)
#             pad[: tile.shape[0], : tile.shape[1]] = tile
#             tile = pad

#         t = torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0) / 255.0
#         t = t.to(device, non_blocking=True)

#         with autocast_ctx:
#             logits = model(t)
#             prob = torch.sigmoid(logits)[0, 0].float().detach().cpu().numpy()

#         yh = min(H, y + patch)
#         xw = min(W, x + patch)
#         ph = yh - y
#         pw = xw - x
#         pred_sum[y:yh, x:xw] += prob[:ph, :pw]
#         pred_cnt[y:yh, x:xw] += 1.0

#     return pred_sum / np.maximum(pred_cnt, 1e-6)


# # =========================
# # Trainer (with validation pack)
# # =========================
# class SemanticUNetTrainer:
#     """
#     trainer:
#       _target_: dataflow_train.train.SemanticUNetTrainer
#     """

#     def __init__(
#         self,
#         out_dir: str,
#         db_root: str,
#         dataset_roots: Dict[str, str],
#         datasets: List[str],
#         ann_file: str = "ann_semantic.parquet",
#         patch_size: int = 512,
#         epochs: int = 5,
#         train_epoch_size: int = 2000,
#         batch_size: int = 8,
#         num_workers: int = 4,
#         lr: float = 3e-4,
#         wd: float = 1e-4,
#         seed: int = 42,
#         amp: bool = True,
#         log_freq: int = 20,

#         # full-image debug
#         debug_vis_full: int = 4,
#         debug_vis_thr: float = 0.5,
#         debug_vis_max_side: int = 1600,
#         stride: int = 256,

#         # sampling/cache/split
#         pos_fraction: float = 0.7,
#         cache_slides: int = 64,
#         use_meta_split: bool = True,
#         val_ratio: float = 0.1,

#         # ---- validation pack ----
#         sanity_pasteback_n: int = 8,
#         dry_run: bool = False,
#         dry_run_train_steps: int = 50,
#         dry_run_val_steps: int = 20,
#         debug_crop_every: int = 200,
#         debug_crop_max_per_epoch: int = 12,
#         debug_crop_thr: float = 0.5,
#     ):
#         self.out_dir = str(out_dir)
#         self.db_root = str(db_root)
#         self.dataset_roots = dataset_roots
#         self.datasets = datasets
#         self.ann_file = ann_file
#         self.patch_size = int(patch_size)
#         self.epochs = int(epochs)
#         self.train_epoch_size = int(train_epoch_size)
#         self.batch_size = int(batch_size)
#         self.num_workers = int(num_workers)
#         self.lr = float(lr)
#         self.wd = float(wd)
#         self.seed = int(seed)
#         self.amp = bool(amp)
#         self.log_freq = int(log_freq)

#         self.debug_vis_full = int(debug_vis_full)
#         self.debug_vis_thr = float(debug_vis_thr)
#         self.debug_vis_max_side = int(debug_vis_max_side)
#         self.stride = int(stride)

#         self.pos_fraction = float(pos_fraction)
#         self.cache_slides = int(cache_slides)
#         self.use_meta_split = bool(use_meta_split)
#         self.val_ratio = float(val_ratio)

#         self.sanity_pasteback_n = int(sanity_pasteback_n)
#         self.dry_run = bool(dry_run)
#         self.dry_run_train_steps = int(dry_run_train_steps)
#         self.dry_run_val_steps = int(dry_run_val_steps)
#         self.debug_crop_every = int(debug_crop_every)
#         self.debug_crop_max_per_epoch = int(debug_crop_max_per_epoch)
#         self.debug_crop_thr = float(debug_crop_thr)

#         self.rank = int(os.environ.get("RANK", "0"))
#         self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
#         self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

#         random.seed(self.seed + self.rank)
#         np.random.seed(self.seed + self.rank)
#         torch.manual_seed(self.seed + self.rank)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(self.seed + self.rank)
#         torch.backends.cudnn.benchmark = True

#     def _is_main(self) -> bool:
#         return self.rank == 0

#     def _init_dist(self):
#         if self.world_size <= 1:
#             return
#         if torch.distributed.is_initialized():
#             return
#         backend = "nccl" if torch.cuda.is_available() else "gloo"
#         torch.distributed.init_process_group(backend=backend, init_method="env://")
#         if torch.cuda.is_available():
#             torch.cuda.set_device(self.local_rank)

#     def _build_loaders(self):
#         nw = self.num_workers
#         if self.dry_run:
#             nw = 0

#         train_ds = ParquetSemanticCropDataset(
#             db_root=self.db_root,
#             dataset_roots=self.dataset_roots,
#             datasets=self.datasets,
#             split="train",
#             ann_file=self.ann_file,
#             patch_size=self.patch_size,
#             epoch_size=self.train_epoch_size,
#             pos_fraction=self.pos_fraction,
#             seed=self.seed + 123,
#             cache_slides=self.cache_slides,
#             use_meta_split=self.use_meta_split,
#             val_ratio=self.val_ratio,
#         )
#         val_ds = ParquetSemanticCropDataset(
#             db_root=self.db_root,
#             dataset_roots=self.dataset_roots,
#             datasets=self.datasets,
#             split="val",
#             ann_file=self.ann_file,
#             patch_size=self.patch_size,
#             epoch_size=0,
#             pos_fraction=0.0,
#             seed=self.seed + 999,
#             cache_slides=self.cache_slides,
#             use_meta_split=self.use_meta_split,
#             val_ratio=self.val_ratio,
#         )

#         if self.world_size > 1:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(
#                 train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
#             )
#             val_sampler = torch.utils.data.distributed.DistributedSampler(
#                 val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
#             )
#         else:
#             train_sampler = None
#             val_sampler = None

#         train_loader = torch.utils.data.DataLoader(
#             train_ds,
#             batch_size=self.batch_size,
#             shuffle=(train_sampler is None),
#             sampler=train_sampler,
#             num_workers=nw,
#             pin_memory=True,
#             drop_last=True,
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_ds,
#             batch_size=1,
#             shuffle=False,
#             sampler=val_sampler,
#             num_workers=0 if self.dry_run else max(1, nw // 2),
#             pin_memory=True,
#             drop_last=False,
#         )
#         return train_ds, val_ds, train_loader, val_loader

#     def _build_model(self):
#         return UNetSmall(in_ch=3, out_ch=1, base=32)

#     def _save_ckpt(self, model, optim, epoch, best_score, name: str):
#         if not self._is_main():
#             return
#         ckpt = {
#             "epoch": epoch,
#             "best_score": float(best_score),
#             "model": model.state_dict(),
#             "optim": optim.state_dict(),
#         }
#         p = Path(self.out_dir) / "checkpoints"
#         p.mkdir(parents=True, exist_ok=True)
#         torch.save(ckpt, str(p / name))

#     def _sanity_check_pasteback(self, ds: ParquetSemanticCropDataset):
#         """Verify fast crop-mask == slow full-paste then crop."""
#         if not self._is_main():
#             return
#         if self.sanity_pasteback_n <= 0:
#             return
#         if len(ds.slide_uids) == 0:
#             logging.info("[sanity] no slides, skip")
#             return

#         out_dir = Path(self.out_dir) / "sanity_pasteback"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         rng = random.Random(1234)
#         mismatches = 0

#         for k in range(self.sanity_pasteback_n):
#             suid = rng.choice(ds.slide_uids)
#             mr = ds.meta[ds.meta["slide_uid"].astype(str) == suid].iloc[0].to_dict()
#             img_full = ds._load_image_full(mr)
#             H, W = img_full.shape[0], img_full.shape[1]
#             ps = ds.patch_size

#             if H <= ps or W <= ps:
#                 continue

#             # sample random crop
#             cx = rng.randint(0, W - ps)
#             cy = rng.randint(0, H - ps)

#             fast = ds._make_crop_mask_fast(suid, cx, cy, ps)
#             full = ds.build_full_mask_slow(suid, H=H, W=W)
#             slow = full[cy:cy + ps, cx:cx + ps]

#             diff = (fast.astype(np.uint8) ^ slow.astype(np.uint8))
#             n_diff = int(diff.sum())

#             if n_diff != 0:
#                 mismatches += 1
#                 crop = img_full[cy:cy + ps, cx:cx + ps]
#                 fast_bw = np.stack([(fast.astype(np.uint8) * 255)] * 3, axis=-1)
#                 slow_bw = np.stack([(slow.astype(np.uint8) * 255)] * 3, axis=-1)
#                 diff_bw = np.stack([(diff.astype(np.uint8) * 255)] * 3, axis=-1)
#                 panels = [crop, fast_bw, slow_bw, diff_bw]  # img | fast | slow | diff
#                 name = f"{k:02d}_{str(suid).replace('/','_')}_x{cx}_y{cy}_diff{n_diff}.png"
#                 _concat_panels_h(panels, str(out_dir / name), max_side=1600)

#         if mismatches == 0:
#             logging.info(f"[sanity] pasteback OK: {self.sanity_pasteback_n} samples, no mismatch")
#         else:
#             logging.warning(f"[sanity] pasteback mismatch={mismatches}/{self.sanity_pasteback_n}. See {out_dir}")

#     def _debug_full_vis(self, model, val_ds: ParquetSemanticCropDataset, epoch: int, device: torch.device):
#         if not self._is_main() or self.debug_vis_full <= 0:
#             return
#         out_dir = Path(self.out_dir) / "debug_vis_full" / f"epoch{epoch:03d}"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         slide_uids = val_ds.slide_uids
#         if not slide_uids:
#             return

#         pick = slide_uids[: min(self.debug_vis_full, len(slide_uids))]

#         for suid in pick:
#             mr = val_ds.meta[val_ds.meta["slide_uid"].astype(str) == suid].iloc[0].to_dict()
#             img_full = val_ds._load_image_full(mr)
#             H, W = img_full.shape[0], img_full.shape[1]

#             gt_full = val_ds.build_full_mask_slow(suid, H=H, W=W)
#             pred_full = tiled_predict_full(model, img_full, device, patch=self.patch_size, stride=self.stride, amp=self.amp)

#             safe_name = str(suid).replace("/", "_").replace(":", "_")
#             out_path = str(out_dir / f"{safe_name}.png")
#             save_side_by_side_full(
#                 img_u8=img_full,
#                 gt_full=gt_full,
#                 pred_full_prob=pred_full,
#                 out_path=out_path,
#                 thr=self.debug_vis_thr,
#                 max_side=self.debug_vis_max_side,
#             )

#         logging.info(f"[debug_full] saved to: {out_dir}")

#     def _maybe_save_crop_debug(self, epoch: int, global_step: int, batch: Dict[str, Any], prob: torch.Tensor, tag: str):
#         """Save a single crop debug panel: img | gt | pred | error overlay."""
#         if not self._is_main():
#             return
#         if self.debug_crop_every <= 0:
#             return
#         if global_step % self.debug_crop_every != 0:
#             return

#         # limit per-epoch
#         # we maintain a counter file in memory
#         if not hasattr(self, "_debug_crop_count"):
#             self._debug_crop_count = {}
#         c = self._debug_crop_count.get(epoch, 0)
#         if c >= self.debug_crop_max_per_epoch:
#             return
#         self._debug_crop_count[epoch] = c + 1

#         img = batch["image"][0].detach().cpu().float().numpy()  # [3,H,W]
#         gt = batch["mask"][0, 0].detach().cpu().numpy().astype(bool)  # [H,W]
#         pr = prob[0, 0].detach().cpu().float().numpy()  # [H,W]

#         img_u8 = (np.clip(img.transpose(1, 2, 0), 0, 1) * 255.0).astype(np.uint8)

#         suid = str(batch.get("slide_uid", ["unknown"])[0])
#         out_dir = Path(self.out_dir) / "debug_vis_crop" / f"epoch{epoch:03d}"
#         out_dir.mkdir(parents=True, exist_ok=True)
#         name = f"{tag}_step{global_step:07d}_{suid.replace('/','_')}.png"
#         save_side_by_side_crop(img_u8, gt, pr, str(out_dir / name), thr=self.debug_crop_thr, max_side=1600)

#     def run(self):
#         self._init_dist()

#         device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
#         if device.type == "cuda":
#             torch.cuda.set_device(self.local_rank)

#         setup_run_logging(self.out_dir, rank=self.rank)
#         logging.info(f"device={device} rank={self.rank} world={self.world_size}")

#         train_ds, val_ds, train_loader, val_loader = self._build_loaders()

#         # ---- Sanity check paste-back alignment (critical) ----
#         self._sanity_check_pasteback(val_ds)

#         model = self._build_model().to(device)
#         if self.world_size > 1:
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)

#         optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
#         bce = nn.BCEWithLogitsLoss()
#         best = -1.0

#         scaler = torch.cuda.amp.GradScaler(enabled=(self.amp and device.type == "cuda"))

#         global_step = 0

#         for epoch in range(1, self.epochs + 1):
#             if self.world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
#                 train_loader.sampler.set_epoch(epoch)

#             # ---- train ----
#             model.train()
#             t0 = time.time()
#             losses, dices = [], []

#             for it, batch in enumerate(train_loader, start=1):
#                 global_step += 1
#                 img = batch["image"].to(device, non_blocking=True)
#                 msk = batch["mask"].to(device, non_blocking=True)

#                 optim.zero_grad(set_to_none=True)

#                 with torch.cuda.amp.autocast(enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
#                     logits = model(img)
#                     loss = 0.7 * bce(logits, msk) + 0.3 * (1.0 - (2.0 * (torch.sigmoid(logits) * msk).sum(dim=(1,2,3)) + 1e-6) /
#                                                            ((torch.sigmoid(logits)).sum(dim=(1,2,3)) + msk.sum(dim=(1,2,3)) + 1e-6)).mean()

#                 scaler.scale(loss).backward()
#                 scaler.step(optim)
#                 scaler.update()

#                 with torch.no_grad():
#                     prob = torch.sigmoid(logits)
#                     # batch dice (hard, threshold 0.5) for quick monitoring
#                     pred = (prob > 0.5).float()
#                     inter = (pred * msk).sum(dim=(1,2,3))
#                     den = pred.sum(dim=(1,2,3)) + msk.sum(dim=(1,2,3))
#                     d = ((2 * inter + 1e-6) / (den + 1e-6)).mean().item()

#                 losses.append(float(loss.item()))
#                 dices.append(float(d))

#                 # crop debug (train)
#                 self._maybe_save_crop_debug(epoch, global_step, batch, prob, tag="train")

#                 if self._is_main() and (it % self.log_freq == 0):
#                     logging.info(
#                         f"Epoch {epoch:03d} | it {it:04d}/{len(train_loader)} "
#                         f"loss {np.mean(losses):.4f} dice {np.mean(dices):.4f}"
#                     )

#                 # dry-run break
#                 if self.dry_run and it >= self.dry_run_train_steps:
#                     if self._is_main():
#                         logging.info(f"[dry_run] stop train at it={it}")
#                     break

#             train_loss = float(np.mean(losses)) if losses else 0.0
#             train_dice = float(np.mean(dices)) if dices else 0.0

#             # ---- val (stronger stats) ----
#             model.eval()
#             v_losses = []
#             dice_all_list: List[float] = []
#             dice_nonempty_list: List[float] = []
#             empty_gt = 0
#             empty_pred = 0
#             fg_frac_list: List[float] = []

#             with torch.no_grad():
#                 for vi, batch in enumerate(val_loader, start=1):
#                     img = batch["image"].to(device, non_blocking=True)
#                     msk = batch["mask"].to(device, non_blocking=True)

#                     with torch.cuda.amp.autocast(enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
#                         logits = model(img)
#                         loss = 0.7 * bce(logits, msk) + 0.3 * (1.0 - (2.0 * (torch.sigmoid(logits) * msk).sum(dim=(1,2,3)) + 1e-6) /
#                                                                ((torch.sigmoid(logits)).sum(dim=(1,2,3)) + msk.sum(dim=(1,2,3)) + 1e-6)).mean()
#                     v_losses.append(float(loss.item()))

#                     prob = torch.sigmoid(logits)[0, 0].float().detach().cpu().numpy()
#                     gt = msk[0, 0].detach().cpu().numpy().astype(bool)

#                     fg = float(gt.mean())
#                     fg_frac_list.append(fg)

#                     d = dice_per_image_from_prob(prob, gt, thr=self.debug_crop_thr)
#                     dice_all_list.append(d)

#                     if gt.sum() == 0:
#                         empty_gt += 1
#                         if (prob >= self.debug_crop_thr).sum() == 0:
#                             empty_pred += 1
#                     else:
#                         dice_nonempty_list.append(d)

#                     # crop debug (val) —— 用 val 的第一个 sample/或按间隔抽样也行
#                     if self._is_main() and (vi == 1 or (self.debug_crop_every > 0 and (vi % max(1, self.debug_crop_every // 50) == 0))):
#                         # fabricate a "global_step" like id for val saving
#                         self._maybe_save_crop_debug(epoch, global_step + vi, batch, torch.from_numpy(prob[None, None]).to(device), tag="val")

#                     if self.dry_run and vi >= self.dry_run_val_steps:
#                         if self._is_main():
#                             logging.info(f"[dry_run] stop val at vi={vi}")
#                         break

#             val_loss = float(np.mean(v_losses)) if v_losses else 0.0
#             dice_all = summarize_array(dice_all_list)
#             dice_nonempty = summarize_array(dice_nonempty_list)
#             empty_gt_rate = float(empty_gt / max(1, len(dice_all_list)))
#             empty_pred_rate = float(empty_pred / max(1, empty_gt)) if empty_gt > 0 else 0.0
#             fg_frac = summarize_array(fg_frac_list)

#             # reduce across ranks (coarse: only mean values; lists not reduced)
#             if self.world_size > 1:
#                 tl = torch.tensor([train_loss, train_dice, val_loss, dice_all["mean"], dice_nonempty["mean"], empty_gt_rate], device=device)
#                 torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
#                 tl = tl / float(self.world_size)
#                 train_loss, train_dice, val_loss, dice_all_mean, dice_nonempty_mean, empty_gt_rate_mean = [float(x) for x in tl.tolist()]
#             else:
#                 dice_all_mean = dice_all["mean"]
#                 dice_nonempty_mean = dice_nonempty["mean"]
#                 empty_gt_rate_mean = empty_gt_rate

#             if self._is_main():
#                 dt = time.time() - t0
#                 logging.info(
#                     f"Epoch {epoch:03d} | train loss {train_loss:.4f} dice {train_dice:.4f} "
#                     f"| val loss {val_loss:.4f} "
#                     f"| dice_all mean {dice_all_mean:.4f} (p50 {dice_all['p50']:.4f}, p90 {dice_all['p90']:.4f}) "
#                     f"| dice_nonempty mean {dice_nonempty_mean:.4f} (p50 {dice_nonempty['p50']:.4f}, p90 {dice_nonempty['p90']:.4f}) "
#                     f"| empty_gt_rate {empty_gt_rate_mean:.3f} | fg_frac mean {fg_frac['mean']:.4f} | {dt:.1f}s"
#                 )
#                 if empty_gt > 0:
#                     logging.info(f"          empty_pred_rate_on_empty_gt = {empty_pred_rate:.3f} (how often model predicts empty when GT is empty)")

#             # checkpoints (use dice_nonempty_mean as selection to avoid inflated best)
#             score_for_best = dice_nonempty_mean
#             core = model.module if hasattr(model, "module") else model
#             if score_for_best > best:
#                 best = score_for_best
#                 self._save_ckpt(core, optim, epoch, best, "best.pt")
#             self._save_ckpt(core, optim, epoch, best, "last.pt")

#             # full-image debug vis (optional but very useful)
#             self._debug_full_vis(core, val_ds, epoch, device)

#             if self.dry_run:
#                 if self._is_main():
#                     logging.info("[dry_run] done after 1 epoch")
#                 break

#         if self._is_main():
#             logging.info(f"[DONE] best_score(dice_nonempty_mean)={best:.4f} out_dir={self.out_dir}")


# # =========================
# # Launchers (local spawn + submitit)
# # =========================
# os.environ["HYDRA_FULL_ERROR"] = "1"


# def single_proc_run(local_rank: int, main_port: int, cfg, world_size: int):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(main_port)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     os.environ["WORLD_SIZE"] = str(world_size)

#     add_pythonpath_to_sys_path()

#     exp_dir = cfg.launcher.experiment_log_dir
#     if exp_dir is None:
#         exp_dir = str(Path(os.getcwd()) / "runs" / "unnamed")
#         cfg.launcher.experiment_log_dir = exp_dir

#     if "out_dir" not in cfg.trainer:
#         cfg.trainer.out_dir = exp_dir

#     trainer = instantiate(cfg.trainer, _recursive_=False)
#     trainer.run()


# def single_node_runner(cfg, main_port: int):
#     assert int(cfg.launcher.num_nodes) == 1
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
#             exp_dir = self.cfg.launcher.experiment_log_dir
#             if "out_dir" not in self.cfg.trainer:
#                 self.cfg.trainer.out_dir = exp_dir
#             trainer = instantiate(self.cfg.trainer, _recursive_=False)
#             trainer.run()
#         except Exception as e:
#             logging.error(format_exception(e))
#             raise


# def dump_cfg_files(cfg, exp_dir: str):
#     exp = Path(exp_dir)
#     exp.mkdir(parents=True, exist_ok=True)

#     with open(exp / "config.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg))

#     resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
#     with open(exp / "config_resolved.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(resolved))


# def _ensure_cfg_defaults(cfg, args):
#     """
#     Make cfg robust even if some top-level groups (launcher/submitit/trainer) are missing.
#     Also applies CLI overrides in a struct-safe way.
#     """
#     # 1) ensure groups exist
#     if OmegaConf.select(cfg, "launcher") is None:
#         OmegaConf.update(cfg, "launcher", {}, force_add=True)
#     if OmegaConf.select(cfg, "submitit") is None:
#         OmegaConf.update(cfg, "submitit", {}, force_add=True)
#     if OmegaConf.select(cfg, "trainer") is None:
#         OmegaConf.update(cfg, "trainer", {}, force_add=True)

#     # 2) defaults (only fill if missing)
#     def sel(path, default=None):
#         v = OmegaConf.select(cfg, path)
#         return default if v is None else v

#     OmegaConf.update(cfg, "launcher.num_nodes", int(sel("launcher.num_nodes", 1)), force_add=True)
#     OmegaConf.update(cfg, "launcher.gpus_per_node", int(sel("launcher.gpus_per_node", 1)), force_add=True)
#     OmegaConf.update(cfg, "launcher.experiment_log_dir", sel("launcher.experiment_log_dir", None), force_add=True)

#     OmegaConf.update(cfg, "submitit.use_cluster", bool(sel("submitit.use_cluster", False)), force_add=True)
#     OmegaConf.update(cfg, "submitit.timeout_hour", int(sel("submitit.timeout_hour", 24)), force_add=True)
#     OmegaConf.update(cfg, "submitit.cpus_per_task", int(sel("submitit.cpus_per_task", 8)), force_add=True)
#     OmegaConf.update(cfg, "submitit.partition", sel("submitit.partition", None), force_add=True)
#     OmegaConf.update(cfg, "submitit.name", str(sel("submitit.name", getattr(args, "config", "exp"))), force_add=True)
#     OmegaConf.update(cfg, "submitit.port_range", sel("submitit.port_range", [15000, 19999]), force_add=True)

#     # 3) CLI overrides (struct-safe)
#     if args.num_gpus is not None:
#         OmegaConf.update(cfg, "launcher.gpus_per_node", int(args.num_gpus), force_add=True)
#     if args.num_nodes is not None:
#         OmegaConf.update(cfg, "launcher.num_nodes", int(args.num_nodes), force_add=True)
#     if args.use_cluster is not None:
#         OmegaConf.update(cfg, "submitit.use_cluster", bool(args.use_cluster), force_add=True)
#     if args.dry_run is not None:
#         OmegaConf.update(cfg, "trainer.dry_run", bool(args.dry_run), force_add=True)

#     return cfg


# # def main(args):
# #     cfg = compose(config_name=args.config)

# #     # cmdline overrides
# #     if args.num_gpus is not None:
# #         cfg.launcher.gpus_per_node = int(args.num_gpus)
# #     if args.num_nodes is not None:
# #         cfg.launcher.num_nodes = int(args.num_nodes)
# #     if args.use_cluster is not None:
# #         cfg.submitit.use_cluster = bool(args.use_cluster)
# #     if args.dry_run is not None:
# #         cfg.trainer.dry_run = bool(args.dry_run)
# #         #OmegaConf.update(cfg, "trainer.dry_run", bool(args.dry_run), force_add=True)

# #     # exp dir (append timestamp)
# #     exp_dir = cfg.launcher.experiment_log_dir
# #     if exp_dir is None:
# #         exp_dir = str(Path(os.getcwd()) / "runs" / f"{args.config}_{now_ts()}")
# #         cfg.launcher.experiment_log_dir = exp_dir
# #     else:
# #         exp_dir = str(Path(exp_dir) / now_ts())
# #         cfg.launcher.experiment_log_dir = exp_dir
# #     makedir(exp_dir)

# #     print("###################### Train App Config ####################")
# #     print(OmegaConf.to_yaml(cfg))
# #     print("############################################################")

# #     dump_cfg_files(cfg, exp_dir)
# #     add_pythonpath_to_sys_path()

# #     if bool(cfg.submitit.use_cluster):
# #         submitit_dir = str(Path(exp_dir) / "submitit_logs")
# #         executor = submitit.AutoExecutor(folder=submitit_dir)

# #         job_kwargs = {
# #             "timeout_min": int(cfg.submitit.timeout_hour) * 60,
# #             "name": str(cfg.submitit.name),
# #             "slurm_partition": None if cfg.submitit.partition in (None, "null") else str(cfg.submitit.partition),
# #             "gpus_per_node": int(cfg.launcher.gpus_per_node),
# #             "tasks_per_node": int(cfg.launcher.gpus_per_node),
# #             "cpus_per_task": int(cfg.submitit.cpus_per_task),
# #             "nodes": int(cfg.launcher.num_nodes),
# #         }
# #         executor.update_parameters(**job_kwargs)

# #         port = random.randint(int(cfg.submitit.port_range[0]), int(cfg.submitit.port_range[1]))
# #         runner = SubmititRunner(port, cfg)
# #         job = executor.submit(runner)
# #         print("Submitit Job ID:", job.job_id)
# #     else:
# #         cfg.launcher.num_nodes = 1
# #         port = random.randint(int(cfg.submitit.port_range[0]), int(cfg.submitit.port_range[1]))
# #         single_node_runner(cfg, port)


# def main(args):
#     cfg = compose(config_name=args.config)

#     # 🔍 先打印一下实际加载到的 cfg（用于定位“为什么 launcher 不见了”）
#     print("###################### Loaded Config (raw) ######################")
#     try:
#         print(OmegaConf.to_yaml(cfg))
#     except Exception as e:
#         print("Failed to print cfg yaml:", e)
#         print("cfg keys:", list(cfg.keys()) if hasattr(cfg, "keys") else type(cfg))
#     print("#################################################################")

#     cfg = _ensure_cfg_defaults(cfg, args)

#     # exp dir: if not set, create one
#     exp_dir = OmegaConf.select(cfg, "launcher.experiment_log_dir")
#     if exp_dir is None:
#         exp_dir = str(Path(os.getcwd()) / "runs" / f"{args.config}_{now_ts()}")
#     else:
#         exp_dir = str(Path(exp_dir) / now_ts())
#     OmegaConf.update(cfg, "launcher.experiment_log_dir", exp_dir, force_add=True)

#     makedir(exp_dir)

#     print("###################### Train App Config (final) ##################")
#     print(OmegaConf.to_yaml(cfg))
#     print("#################################################################")

#     dump_cfg_files(cfg, exp_dir)
#     add_pythonpath_to_sys_path()

#     # launch
#     use_cluster = bool(OmegaConf.select(cfg, "submitit.use_cluster"))
#     if use_cluster:
#         submitit_dir = str(Path(exp_dir) / "submitit_logs")
#         executor = submitit.AutoExecutor(folder=submitit_dir)

#         port_range = OmegaConf.select(cfg, "submitit.port_range") or [15000, 19999]
#         port = random.randint(int(port_range[0]), int(port_range[1]))

#         job_kwargs = {
#             "timeout_min": int(OmegaConf.select(cfg, "submitit.timeout_hour")) * 60,
#             "name": str(OmegaConf.select(cfg, "submitit.name")),
#             "slurm_partition": OmegaConf.select(cfg, "submitit.partition"),
#             "gpus_per_node": int(OmegaConf.select(cfg, "launcher.gpus_per_node")),
#             "tasks_per_node": int(OmegaConf.select(cfg, "launcher.gpus_per_node")),
#             "cpus_per_task": int(OmegaConf.select(cfg, "submitit.cpus_per_task")),
#             "nodes": int(OmegaConf.select(cfg, "launcher.num_nodes")),
#         }
#         executor.update_parameters(**job_kwargs)

#         runner = SubmititRunner(port, cfg)
#         job = executor.submit(runner)
#         print("Submitit Job ID:", job.job_id)
#     else:
#         OmegaConf.update(cfg, "launcher.num_nodes", 1, force_add=True)
#         port_range = OmegaConf.select(cfg, "submitit.port_range") or [15000, 19999]
#         port = random.randint(int(port_range[0]), int(port_range[1]))
#         single_node_runner(cfg, port)




# if __name__ == "__main__":
#     # configs live at: dataflow_train/configs/<name>.yaml
#     initialize_config_module("dataflow_train.configs", version_base="1.2")

#     parser = ArgumentParser()
#     parser.add_argument("-c", "--config", required=True, type=str, help="config name under dataflow_train/configs")
#     parser.add_argument("--use-cluster", type=int, default=None, help="0 local / 1 slurm")
#     parser.add_argument("--num-gpus", type=int, default=None)
#     parser.add_argument("--num-nodes", type=int, default=None)
#     parser.add_argument("--dry-run", type=int, default=None, help="1: run quick validation (1 epoch, limited steps)")
#     args = parser.parse_args()
#     args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

#     main(args)





# # dataflow_train/train.py
# from __future__ import annotations

# import os
# import sys
# import time
# import json
# import math
# import random
# import logging
# import traceback
# import warnings
# from dataclasses import dataclass
# from argparse import ArgumentParser
# from pathlib import Path
# from collections import OrderedDict
# from typing import Any, Dict, List, Tuple, Optional

# import numpy as np
# import pandas as pd
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.amp import autocast, GradScaler

# import submitit
# from hydra import compose, initialize_config_module
# from hydra.utils import instantiate
# from omegaconf import OmegaConf

# # ----------------------------
# # Global flags
# # ----------------------------
# os.environ.setdefault("HYDRA_FULL_ERROR", "1")
# warnings.filterwarnings("ignore", category=FutureWarning)  # 已经迁移到 torch.amp，正常不会刷屏


# # =========================
# # Small helpers
# # =========================
# def now_ts() -> str:
#     return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


# def makedir(p: str | Path):
#     Path(p).mkdir(parents=True, exist_ok=True)


# def add_pythonpath_to_sys_path():
#     pp = os.environ.get("PYTHONPATH", "")
#     if not pp:
#         return
#     for x in pp.split(":"):
#         if x and x not in sys.path:
#             sys.path.insert(0, x)


# def format_exception(e: Exception, limit: int = 40) -> str:
#     tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
#     return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


# # =========================
# # Logging (stdout/stderr tee)
# # =========================
# class Tee:
#     def __init__(self, *streams):
#         self.streams = streams

#     def write(self, data):
#         for s in self.streams:
#             try:
#                 s.write(data)
#                 s.flush()
#             except Exception:
#                 pass

#     def flush(self):
#         for s in self.streams:
#             try:
#                 s.flush()
#             except Exception:
#                 pass


# def setup_run_logging(out_dir: str, rank: int = 0) -> str:
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     log_path = out_dir / ("train.log" if rank == 0 else f"train_rank{rank}.log")

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[
#             logging.FileHandler(log_path, mode="a", encoding="utf-8"),
#             logging.StreamHandler(sys.__stdout__),
#         ],
#     )

#     f = open(log_path, "a", encoding="utf-8", buffering=1)
#     sys.stdout = Tee(sys.__stdout__, f)
#     sys.stderr = Tee(sys.__stderr__, f)

#     logging.info(f"[rank{rank}] Logging to: {log_path}")
#     return str(log_path)


# def dump_cfg_files(cfg, exp_dir: str):
#     exp_dir = Path(exp_dir)
#     exp_dir.mkdir(parents=True, exist_ok=True)

#     with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg))

#     # resolved
#     cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
#     with open(exp_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg_resolved))


# # =========================
# # RLE decode (COCO counts, Fortran order)
# # =========================
# def _to_counts_list(counts: Any) -> List[int]:
#     if counts is None:
#         return []
#     if isinstance(counts, (list, tuple)):
#         return [int(x) for x in counts]
#     if isinstance(counts, np.ndarray):
#         return [int(x) for x in counts.tolist()]
#     # sometimes parquet stores as string like "[1,2,3]" (rare)
#     if isinstance(counts, str):
#         s = counts.strip()
#         if s.startswith("[") and s.endswith("]"):
#             try:
#                 return [int(x) for x in json.loads(s)]
#             except Exception:
#                 pass
#         # fallback split
#         return [int(x) for x in s.replace(",", " ").split()]
#     # unknown
#     return [int(counts)]


# def rle_decode_counts(counts: Any, h: int, w: int) -> np.ndarray:
#     """
#     COCO-style RLE counts, Fortran order, counts start with zeros.
#     Returns bool mask [h,w].
#     """
#     counts = _to_counts_list(counts)
#     total = int(h) * int(w)
#     if not counts:
#         return np.zeros((h, w), dtype=bool)
#     s = int(sum(counts))
#     if s != total:
#         raise ValueError(f"sum(counts)={s} != h*w={total} (h={h},w={w})")

#     flat = np.zeros(total, dtype=np.uint8)
#     idx = 0
#     val = 0
#     for run in counts:
#         run = int(run)
#         if val == 1 and run > 0:
#             flat[idx : idx + run] = 1
#         idx += run
#         val ^= 1
#     return flat.reshape((h, w), order="F").astype(bool)


# # =========================
# # Visualization utils
# # =========================
# def tensor_to_uint8_img(x) -> np.ndarray:
#     if torch.is_tensor(x):
#         x = x.detach().cpu().float()
#         if x.ndim == 3 and x.shape[0] in (1, 3):
#             x = x.permute(1, 2, 0)
#         x = x.numpy()

#     if x.ndim == 2:
#         x = np.stack([x, x, x], axis=-1)

#     if x.max() <= 1.5:
#         x = x * 255.0
#     x = np.clip(x, 0, 255).astype(np.uint8)
#     return x


# def overlay_mask(img_uint8: np.ndarray, mask_bool: np.ndarray, color=(255, 0, 0), alpha=0.55) -> np.ndarray:
#     img = img_uint8.astype(np.float32).copy()
#     m = mask_bool.astype(bool)
#     if m.ndim == 2:
#         m = m[..., None]
#     color_arr = np.array(color, dtype=np.float32)[None, None, :]
#     img[m[..., 0]] = img[m[..., 0]] * (1 - alpha) + color_arr * alpha
#     return np.clip(img, 0, 255).astype(np.uint8)


# def save_side_by_side(
#     img_u8: np.ndarray,
#     gt: np.ndarray,
#     pred_prob: np.ndarray,
#     out_path: str,
#     thr: float = 0.5,
#     max_side: int = 1600,
# ):
#     gt = gt.astype(bool)
#     pred = (pred_prob >= thr)

#     gt_vis = overlay_mask(img_u8, gt, color=(0, 255, 0), alpha=0.55)
#     pr_vis = overlay_mask(img_u8, pred, color=(255, 0, 0), alpha=0.55)

#     gt_bw = np.stack([(gt.astype(np.uint8) * 255)] * 3, axis=-1)
#     pr_bw = np.stack([(pred.astype(np.uint8) * 255)] * 3, axis=-1)

#     panels = [img_u8, gt_vis, pr_vis, gt_bw, pr_bw]
#     panel_imgs = [Image.fromarray(p) for p in panels]

#     def downscale(im: Image.Image) -> Image.Image:
#         w, h = im.size
#         s = max(w, h)
#         if s <= max_side:
#             return im
#         scale = max_side / float(s)
#         nw, nh = int(w * scale), int(h * scale)
#         return im.resize((nw, nh), Image.BILINEAR)

#     panel_imgs = [downscale(im) for im in panel_imgs]

#     W = sum(im.size[0] for im in panel_imgs)
#     H = max(im.size[1] for im in panel_imgs)
#     canvas = Image.new("RGB", (W, H))
#     x = 0
#     for im in panel_imgs:
#         canvas.paste(im, (x, 0))
#         x += im.size[0]

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     canvas.save(out_path)


# # =========================
# # Minimal UNet
# # =========================
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class UNetSmall(nn.Module):
#     def __init__(self, in_ch=3, out_ch=1, base=32):
#         super().__init__()
#         self.enc1 = ConvBlock(in_ch, base)
#         self.pool1 = nn.MaxPool2d(2)
#         self.enc2 = ConvBlock(base, base * 2)
#         self.pool2 = nn.MaxPool2d(2)
#         self.enc3 = ConvBlock(base * 2, base * 4)
#         self.pool3 = nn.MaxPool2d(2)

#         self.mid = ConvBlock(base * 4, base * 8)

#         self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
#         self.dec3 = ConvBlock(base * 8, base * 4)
#         self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
#         self.dec2 = ConvBlock(base * 4, base * 2)
#         self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
#         self.dec1 = ConvBlock(base * 2, base)

#         self.out = nn.Conv2d(base, out_ch, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))
#         m = self.mid(self.pool3(e3))

#         d3 = self.up3(m)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))
#         d2 = self.up2(d3)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))
#         d1 = self.up1(d2)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))
#         return self.out(d1)


# # =========================
# # Loss / metrics
# # =========================
# def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
#     prob = torch.sigmoid(logits)
#     inter = (prob * target).sum(dim=(1, 2, 3))
#     den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
#     dice = (2 * inter + eps) / (den + eps)
#     return 1 - dice.mean()


# @torch.no_grad()
# def dice_per_sample(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps=1e-6) -> torch.Tensor:
#     """
#     prob: [B,1,H,W] in [0,1]
#     target: [B,1,H,W] float {0,1}
#     returns dice [B]
#     """
#     pred = (prob > thr).float()
#     inter = (pred * target).sum(dim=(1, 2, 3))
#     den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
#     return (2 * inter + eps) / (den + eps)


# # =========================
# # Parquet dataset (semantic) with ROI paste-back for crop
# # =========================
# def _safe_relpath_join(root: Path, rel_path: str) -> Path:
#     return (root / rel_path).resolve()


# def _norm_split(s: Any) -> str:
#     s = str(s).lower()
#     if s in ("val", "valid", "validation"):
#         return "val"
#     if s in ("train", "tr"):
#         return "train"
#     if s in ("test", "te"):
#         return "test"
#     return s


# class ParquetSemanticCropDataset(torch.utils.data.Dataset):
#     """
#     - Read meta.parquet and ann_semantic.parquet from db_root/<dataset>/
#     - For each slide: annotations are ROI-bbox + RLE over ROI
#     - For crop: build crop mask by intersecting ROI with crop region (equivalent to paste ROI to full canvas then crop)
#     """

#     def __init__(
#         self,
#         db_root: str,
#         dataset_roots: Dict[str, str],
#         datasets: List[str],
#         split: str,
#         ann_file: str = "ann_semantic.parquet",
#         patch_size: int = 512,
#         epoch_size: int = 2000,
#         max_slides: int = 0,
#         pos_fraction: float = 0.7,
#         seed: int = 42,
#         cache_slides: int = 64,
#         use_meta_split: bool = True,
#         val_ratio: float = 0.1,
#     ):
#         super().__init__()
#         self.db_root = Path(db_root)
#         self.dataset_roots = {k.lower(): Path(v) for k, v in dataset_roots.items()}
#         self.datasets = [d.lower() for d in datasets]
#         self.split = _norm_split(split)
#         self.ann_file = ann_file
#         self.patch_size = int(patch_size)
#         self.epoch_size = int(epoch_size)
#         self.pos_fraction = float(pos_fraction)
#         self.rng = random.Random(seed)
#         self.cache_slides = int(cache_slides)
#         self.use_meta_split = bool(use_meta_split)
#         self.val_ratio = float(val_ratio)

#         meta_list = []
#         ann_list = []

#         for ds in self.datasets:
#             meta_path = self.db_root / ds / "meta.parquet"
#             ann_path = self.db_root / ds / ann_file
#             if not meta_path.exists():
#                 raise FileNotFoundError(f"Missing: {meta_path}")
#             if not ann_path.exists():
#                 raise FileNotFoundError(f"Missing: {ann_path}")

#             # ---- meta ----
#             meta = pd.read_parquet(meta_path)
#             # ensure required cols
#             if "slide_uid" not in meta.columns:
#                 raise KeyError(f"{meta_path} missing slide_uid")
#             if "rel_path" not in meta.columns:
#                 raise KeyError(f"{meta_path} missing rel_path")

#             if "dataset" not in meta.columns:
#                 meta["dataset"] = ds

#             # unify H/W columns
#             if "height_px" not in meta.columns and "H" in meta.columns:
#                 meta["height_px"] = meta["H"]
#             if "width_px" not in meta.columns and "W" in meta.columns:
#                 meta["width_px"] = meta["W"]
#             if "height_px" not in meta.columns or "width_px" not in meta.columns:
#                 raise KeyError(f"{meta_path} missing height_px/width_px")

#             # split handling
#             if "split" in meta.columns:
#                 meta["split"] = meta["split"].map(_norm_split)
#             else:
#                 meta["split"] = "unknown"

#             meta_list.append(meta[["slide_uid", "dataset", "rel_path", "width_px", "height_px", "split"]].copy())

#             # ---- ann ----
#             ann = pd.read_parquet(ann_path)
#             # required
#             if "slide_uid" not in ann.columns:
#                 raise KeyError(f"{ann_path} missing slide_uid")

#             # unify roi cols
#             for need in ["roi_x", "roi_y", "roi_w", "roi_h", "rle_size_h", "rle_size_w", "rle_counts"]:
#                 if need not in ann.columns:
#                     raise KeyError(f"{ann_path} missing {need}")

#             ann_list.append(
#                 ann[["slide_uid", "roi_x", "roi_y", "roi_w", "roi_h", "rle_size_h", "rle_size_w", "rle_counts"]].copy()
#             )

#         meta_all = pd.concat(meta_list, ignore_index=True)
#         ann_all = pd.concat(ann_list, ignore_index=True)

#         # split selection
#         if self.use_meta_split and (meta_all["split"] != "unknown").any():
#             meta_all["split"] = meta_all["split"].map(_norm_split)
#             meta_sel = meta_all[meta_all["split"] == self.split].reset_index(drop=True)
#         else:
#             # deterministic split by hash(slide_uid)
#             uids = meta_all["slide_uid"].astype(str).tolist()
#             def uid_to_u01(uid: str) -> float:
#                 # stable-ish hash to [0,1)
#                 h = 2166136261
#                 for ch in uid:
#                     h ^= ord(ch)
#                     h = (h * 16777619) & 0xFFFFFFFF
#                 return (h % 1000000) / 1000000.0

#             split_tags = []
#             for uid in uids:
#                 u = uid_to_u01(uid)
#                 if u < self.val_ratio:
#                     split_tags.append("val")
#                 else:
#                     split_tags.append("train")
#             meta_all = meta_all.copy()
#             meta_all["split"] = split_tags
#             meta_sel = meta_all[meta_all["split"] == self.split].reset_index(drop=True)

#         if max_slides and max_slides > 0:
#             meta_sel = meta_sel.head(int(max_slides)).reset_index(drop=True)

#         self.meta = meta_sel
#         self.slide_uids = self.meta["slide_uid"].astype(str).tolist()

#         # group annotations per slide_uid
#         self.ann_by_slide: Dict[str, List[Dict[str, Any]]] = {}
#         for r in ann_all.to_dict("records"):
#             suid = str(r["slide_uid"])
#             self.ann_by_slide.setdefault(suid, []).append(r)

#         # LRU cache: slide_uid -> decoded ROI masks [(rx,ry,mask_bool,mw,mh)]
#         self._slide_cache: "OrderedDict[str, List[Tuple[int,int,np.ndarray,int,int]]]" = OrderedDict()

#         logging.info(f"[{self.split}] slides={len(self.slide_uids)} ann_file={ann_file} patch={self.patch_size}")

#     def __len__(self):
#         return self.epoch_size if self.split == "train" else len(self.slide_uids)

#     def _load_image_full(self, meta_row: Dict[str, Any]) -> np.ndarray:
#         ds = str(meta_row["dataset"]).lower()
#         root = self.dataset_roots.get(ds)
#         if root is None:
#             raise KeyError(f"dataset_roots missing key: {ds}")
#         p = _safe_relpath_join(root, str(meta_row["rel_path"]))
#         try:
#             with Image.open(p) as im:
#                 im = im.convert("RGB")
#                 return np.array(im)
#         except Exception as e:
#             raise RuntimeError(f"Failed to open image: {p}") from e

#     def _get_slide_decoded_rois(self, slide_uid: str) -> List[Tuple[int, int, np.ndarray, int, int]]:
#         if slide_uid in self._slide_cache:
#             self._slide_cache.move_to_end(slide_uid)
#             return self._slide_cache[slide_uid]

#         anns = self.ann_by_slide.get(slide_uid, [])
#         decoded: List[Tuple[int, int, np.ndarray, int, int]] = []
#         for a in anns:
#             rx, ry = int(a["roi_x"]), int(a["roi_y"])
#             h, w = int(a["rle_size_h"]), int(a["rle_size_w"])
#             counts = a["rle_counts"]
#             if counts is None:
#                 continue
#             m = rle_decode_counts(counts, h, w)
#             decoded.append((rx, ry, m, w, h))

#         self._slide_cache[slide_uid] = decoded
#         self._slide_cache.move_to_end(slide_uid)
#         while len(self._slide_cache) > self.cache_slides:
#             self._slide_cache.popitem(last=False)
#         return decoded

#     def _make_crop_mask(self, slide_uid: str, cx: int, cy: int, ps: int) -> np.ndarray:
#         crop_mask = np.zeros((ps, ps), dtype=bool)
#         x0, y0 = int(cx), int(cy)
#         x1, y1 = x0 + ps, y0 + ps

#         for (rx, ry, roi_mask, mw, mh) in self._get_slide_decoded_rois(slide_uid):
#             ax0, ay0 = rx, ry
#             ax1, ay1 = rx + mw, ry + mh

#             ix0 = max(x0, ax0)
#             iy0 = max(y0, ay0)
#             ix1 = min(x1, ax1)
#             iy1 = min(y1, ay1)
#             if ix1 <= ix0 or iy1 <= iy0:
#                 continue

#             roi_sx0 = ix0 - ax0
#             roi_sy0 = iy0 - ay0
#             roi_sx1 = ix1 - ax0
#             roi_sy1 = iy1 - ay0
#             sub = roi_mask[roi_sy0:roi_sy1, roi_sx0:roi_sx1]
#             if sub.size == 0:
#                 continue

#             cx0 = ix0 - x0
#             cy0 = iy0 - y0
#             cx1 = cx0 + sub.shape[1]
#             cy1 = cy0 + sub.shape[0]
#             crop_mask[cy0:cy1, cx0:cx1] |= sub

#         return crop_mask

#     def _sample_crop_xy(self, slide_uid: str, H: int, W: int) -> Tuple[int, int]:
#         ps = self.patch_size
#         if H <= ps or W <= ps:
#             return 0, 0

#         if self.split == "train" and (self.rng.random() < self.pos_fraction):
#             rois = self._get_slide_decoded_rois(slide_uid)
#             if rois:
#                 rx, ry, m, mw, mh = self.rng.choice(rois)
#                 xmin = max(0, rx - ps + 1)
#                 xmax = min(W - ps, rx + mw - 1)
#                 ymin = max(0, ry - ps + 1)
#                 ymax = min(H - ps, ry + mh - 1)
#                 if xmin <= xmax and ymin <= ymax:
#                     return self.rng.randint(xmin, xmax), self.rng.randint(ymin, ymax)

#         return self.rng.randint(0, W - ps), self.rng.randint(0, H - ps)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         if self.split == "train":
#             slide_uid = self.slide_uids[idx % len(self.slide_uids)]
#         else:
#             slide_uid = self.slide_uids[idx]

#         # meta row
#         mr = self.meta[self.meta["slide_uid"].astype(str) == slide_uid].iloc[0].to_dict()
#         img_full = self._load_image_full(mr)
#         H, W = img_full.shape[0], img_full.shape[1]
#         ps = self.patch_size

#         if self.split == "train":
#             cx, cy = self._sample_crop_xy(slide_uid, H, W)
#         else:
#             # deterministic center crop for stable val
#             cx = max(0, (W - ps) // 2) if W > ps else 0
#             cy = max(0, (H - ps) // 2) if H > ps else 0

#         crop = img_full[cy:cy + ps, cx:cx + ps]
#         if crop.shape[0] != ps or crop.shape[1] != ps:
#             pad = np.zeros((ps, ps, 3), dtype=np.uint8)
#             pad[: crop.shape[0], : crop.shape[1]] = crop
#             crop = pad

#         crop_mask = self._make_crop_mask(slide_uid, cx, cy, ps).astype(np.uint8)

#         img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
#         m_t = torch.from_numpy(crop_mask)[None, ...].float()

#         return {
#             "image": img_t,
#             "mask": m_t,
#             "slide_uid": slide_uid,
#             "crop_xy": (int(cx), int(cy)),
#             "full_hw": (int(H), int(W)),
#             "dataset": str(mr["dataset"]),
#         }


# # =========================
# # Full-image GT + tiled inference (for debug)
# # =========================
# def build_full_gt_mask(dataset: ParquetSemanticCropDataset, slide_uid: str, H: int, W: int) -> np.ndarray:
#     full = np.zeros((H, W), dtype=bool)
#     for (rx, ry, roi_mask, mw, mh) in dataset._get_slide_decoded_rois(slide_uid):
#         x0, y0 = rx, ry
#         x1 = min(W, x0 + roi_mask.shape[1])
#         y1 = min(H, y0 + roi_mask.shape[0])
#         sub = roi_mask[: y1 - y0, : x1 - x0]
#         full[y0:y1, x0:x1] |= sub
#     return full


# @torch.no_grad()
# def tiled_predict_full(
#     model: nn.Module,
#     img_u8: np.ndarray,
#     device: torch.device,
#     patch: int = 512,
#     stride: int = 256,
#     amp: bool = True,
# ) -> np.ndarray:
#     H, W = img_u8.shape[0], img_u8.shape[1]
#     pred_sum = np.zeros((H, W), dtype=np.float32)
#     pred_cnt = np.zeros((H, W), dtype=np.float32)

#     model.eval()

#     ys = list(range(0, max(1, H - patch + 1), stride))
#     xs = list(range(0, max(1, W - patch + 1), stride))
#     if ys[-1] != H - patch:
#         ys.append(max(0, H - patch))
#     if xs[-1] != W - patch:
#         xs.append(max(0, W - patch))

#     for y in ys:
#         for x in xs:
#             tile = img_u8[y:y + patch, x:x + patch]
#             if tile.shape[0] != patch or tile.shape[1] != patch:
#                 pad = np.zeros((patch, patch, 3), dtype=np.uint8)
#                 pad[: tile.shape[0], : tile.shape[1]] = tile
#                 tile = pad

#             t = torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0) / 255.0
#             t = t.to(device, non_blocking=True)

#             with autocast("cuda", enabled=(amp and device.type == "cuda"), dtype=torch.bfloat16):
#                 logits = model(t)
#                 prob = torch.sigmoid(logits)[0, 0].float().detach().cpu().numpy()

#             yh = min(H, y + patch)
#             xw = min(W, x + patch)
#             ph = yh - y
#             pw = xw - x
#             pred_sum[y:yh, x:xw] += prob[:ph, :pw]
#             pred_cnt[y:yh, x:xw] += 1.0

#     return pred_sum / np.maximum(pred_cnt, 1e-6)


# # =========================
# # Trainer
# # =========================
# class SemanticUNetTrainer:
#     def __init__(
#         self,
#         out_dir: str,
#         db_root: str,
#         dataset_roots: Dict[str, str],
#         datasets: List[str],
#         ann_file: str = "ann_semantic.parquet",
#         patch_size: int = 512,
#         epochs: int = 5,
#         train_epoch_size: int = 2000,
#         batch_size: int = 8,
#         num_workers: int = 4,
#         lr: float = 3e-4,
#         wd: float = 1e-4,
#         seed: int = 42,
#         amp: bool = True,
#         log_freq: int = 20,
#         # debug crop
#         debug_vis_crop: int = 16,
#         debug_vis_crop_thr: float = 0.5,
#         debug_vis_crop_max_side: int = 1600,
#         # debug full
#         debug_vis_full: int = 4,
#         debug_vis_thr: float = 0.5,
#         debug_vis_max_side: int = 1600,
#         stride: int = 256,
#         # sampling/cache
#         pos_fraction: float = 0.7,
#         cache_slides: int = 64,
#         # split
#         use_meta_split: bool = True,
#         val_ratio: float = 0.1,
#         # dry-run
#         dry_run: bool = False,
#         dry_run_train_steps: int = 50,
#         dry_run_val_steps: int = 20,
#     ):
#         self.out_dir = str(out_dir)
#         self.db_root = str(db_root)
#         self.dataset_roots = dataset_roots
#         self.datasets = datasets
#         self.ann_file = ann_file
#         self.patch_size = int(patch_size)

#         self.epochs = int(epochs)
#         self.train_epoch_size = int(train_epoch_size)
#         self.batch_size = int(batch_size)
#         self.num_workers = int(num_workers)

#         self.lr = float(lr)
#         self.wd = float(wd)
#         self.seed = int(seed)
#         self.amp = bool(amp)
#         self.log_freq = int(log_freq)

#         self.debug_vis_crop = int(debug_vis_crop)
#         self.debug_vis_crop_thr = float(debug_vis_crop_thr)
#         self.debug_vis_crop_max_side = int(debug_vis_crop_max_side)

#         self.debug_vis_full = int(debug_vis_full)
#         self.debug_vis_thr = float(debug_vis_thr)
#         self.debug_vis_max_side = int(debug_vis_max_side)
#         self.stride = int(stride)

#         self.pos_fraction = float(pos_fraction)
#         self.cache_slides = int(cache_slides)

#         self.use_meta_split = bool(use_meta_split)
#         self.val_ratio = float(val_ratio)

#         self.dry_run = bool(dry_run)
#         self.dry_run_train_steps = int(dry_run_train_steps)
#         self.dry_run_val_steps = int(dry_run_val_steps)

#         self.rank = int(os.environ.get("RANK", "0"))
#         self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
#         self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

#         random.seed(self.seed + self.rank)
#         np.random.seed(self.seed + self.rank)
#         torch.manual_seed(self.seed + self.rank)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(self.seed + self.rank)

#     def _is_main(self) -> bool:
#         return self.rank == 0

#     def _init_dist(self):
#         if self.world_size <= 1:
#             return
#         if torch.distributed.is_initialized():
#             return
#         torch.distributed.init_process_group(backend="nccl", init_method="env://")
#         torch.cuda.set_device(self.local_rank)

#     def _build_loaders(self):
#         # 🔥 dry-run 时禁用多进程 worker，避免提前 break 时随机 abort
#         nw = self.num_workers
#         if self.dry_run:
#             nw = 0

#         train_ds = ParquetSemanticCropDataset(
#             db_root=self.db_root,
#             dataset_roots=self.dataset_roots,
#             datasets=self.datasets,
#             split="train",
#             ann_file=self.ann_file,
#             patch_size=self.patch_size,
#             epoch_size=self.train_epoch_size,
#             pos_fraction=self.pos_fraction,
#             seed=self.seed + 123,
#             cache_slides=self.cache_slides,
#             use_meta_split=self.use_meta_split,
#             val_ratio=self.val_ratio,
#         )
#         val_ds = ParquetSemanticCropDataset(
#             db_root=self.db_root,
#             dataset_roots=self.dataset_roots,
#             datasets=self.datasets,
#             split="val",
#             ann_file=self.ann_file,
#             patch_size=self.patch_size,
#             epoch_size=0,
#             pos_fraction=0.0,
#             seed=self.seed + 999,
#             cache_slides=self.cache_slides,
#             use_meta_split=self.use_meta_split,
#             val_ratio=self.val_ratio,
#         )

#         if self.world_size > 1:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(
#                 train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
#             )
#             val_sampler = torch.utils.data.distributed.DistributedSampler(
#                 val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
#             )
#         else:
#             train_sampler = None
#             val_sampler = None

#         common = {}
#         if nw > 0:
#             common.update(dict(persistent_workers=True, prefetch_factor=2))

#         train_loader = torch.utils.data.DataLoader(
#             train_ds,
#             batch_size=self.batch_size,
#             shuffle=(train_sampler is None),
#             sampler=train_sampler,
#             num_workers=nw,
#             pin_memory=True,
#             drop_last=True,
#             **common,
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_ds,
#             batch_size=1,
#             shuffle=False,
#             sampler=val_sampler,
#             num_workers=0 if self.dry_run else max(1, nw // 2),
#             pin_memory=True,
#             drop_last=False,
#         )
#         return train_ds, val_ds, train_loader, val_loader

#     def _build_model(self):
#         return UNetSmall(in_ch=3, out_ch=1, base=32)

#     def _save_ckpt(self, model, optim, epoch, best_score, name: str):
#         if not self._is_main():
#             return
#         ckpt = {
#             "epoch": epoch,
#             "best_score": float(best_score),
#             "model": model.state_dict(),
#             "optim": optim.state_dict(),
#         }
#         p = Path(self.out_dir) / "checkpoints"
#         p.mkdir(parents=True, exist_ok=True)
#         torch.save(ckpt, str(p / name))

#     @torch.no_grad()
#     def _sanity_check_pasteback(self, ds: ParquetSemanticCropDataset, n: int = 8):
#         if not self._is_main():
#             return
#         if len(ds.slide_uids) == 0:
#             return
#         n = min(n, len(ds))
#         for i in range(n):
#             sample = ds[i]
#             img = sample["image"]
#             suid = sample["slide_uid"]
#             cx, cy = sample["crop_xy"]
#             H, W = sample["full_hw"]
#             ps = ds.patch_size

#             # build crop via crop-mask method
#             crop_mask1 = sample["mask"][0].cpu().numpy().astype(bool)

#             # build full then crop (slow but for sanity)
#             full = build_full_gt_mask(ds, suid, H=H, W=W)
#             crop_mask2 = full[cy:cy+ps, cx:cx+ps]
#             if crop_mask2.shape != crop_mask1.shape:
#                 pad = np.zeros_like(crop_mask1)
#                 h = min(pad.shape[0], crop_mask2.shape[0])
#                 w = min(pad.shape[1], crop_mask2.shape[1])
#                 pad[:h, :w] = crop_mask2[:h, :w]
#                 crop_mask2 = pad

#             if not np.array_equal(crop_mask1, crop_mask2):
#                 diff = np.logical_xor(crop_mask1, crop_mask2)
#                 raise RuntimeError(f"[sanity] pasteback mismatch on slide={suid}, diff_pixels={diff.sum()}")

#         logging.info(f"[sanity] pasteback OK: {n} samples, no mismatch")

#     @torch.no_grad()
#     def _debug_crop_vis(self, model, val_loader, epoch: int, device: torch.device):
#         if not self._is_main():
#             return
#         if self.debug_vis_crop <= 0:
#             return

#         out_dir = Path(self.out_dir) / "debug_vis_crop" / f"epoch{epoch:03d}"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         model.eval()
#         saved = 0
#         for bi, batch in enumerate(val_loader):
#             img = batch["image"].to(device, non_blocking=True)  # [1,3,H,W]
#             msk = batch["mask"].to(device, non_blocking=True)   # [1,1,H,W]
#             suid = batch["slide_uid"][0]
#             cx, cy = batch["crop_xy"][0][0].item(), batch["crop_xy"][1][0].item() if isinstance(batch["crop_xy"], (list, tuple)) else (0, 0)

#             with autocast("cuda", enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
#                 logits = model(img)
#                 prob = torch.sigmoid(logits).float()

#             img_u8 = tensor_to_uint8_img(img[0])
#             gt = (msk[0, 0] > 0.5).detach().cpu().numpy()
#             pr = prob[0, 0].detach().cpu().numpy()

#             safe = str(suid).replace("/", "_").replace(":", "_")
#             out_path = out_dir / f"{safe}_crop{bi:04d}.png"
#             save_side_by_side(img_u8, gt, pr, str(out_path), thr=self.debug_vis_crop_thr, max_side=self.debug_vis_crop_max_side)

#             saved += 1
#             if saved >= self.debug_vis_crop:
#                 break

#         logging.info(f"[debug_crop] saved {saved} crops to: {out_dir}")

#     @torch.no_grad()
#     def _debug_full_vis(self, model, val_ds: ParquetSemanticCropDataset, epoch: int, device: torch.device):
#         if not self._is_main():
#             return
#         if self.debug_vis_full <= 0:
#             return
#         if len(val_ds.slide_uids) == 0:
#             return

#         out_dir = Path(self.out_dir) / "debug_vis_full" / f"epoch{epoch:03d}"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         pick = val_ds.slide_uids[: min(self.debug_vis_full, len(val_ds.slide_uids))]

#         for suid in pick:
#             mr = val_ds.meta[val_ds.meta["slide_uid"].astype(str) == suid].iloc[0].to_dict()
#             img_full = val_ds._load_image_full(mr)
#             H, W = img_full.shape[0], img_full.shape[1]
#             gt_full = build_full_gt_mask(val_ds, suid, H=H, W=W)

#             pred_full = tiled_predict_full(
#                 model=model,
#                 img_u8=img_full,
#                 device=device,
#                 patch=self.patch_size,
#                 stride=self.stride,
#                 amp=self.amp,
#             )

#             safe = str(suid).replace("/", "_").replace(":", "_")
#             out_path = out_dir / f"{safe}.png"
#             save_side_by_side(img_full, gt_full, pred_full, str(out_path), thr=self.debug_vis_thr, max_side=self.debug_vis_max_side)

#         logging.info(f"[debug_full] saved to: {out_dir}")

#     def run(self):
#         self._init_dist()

#         device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
#         if device.type == "cuda":
#             torch.cuda.set_device(self.local_rank)

#         setup_run_logging(self.out_dir, rank=self.rank)
#         logging.info(f"device={device} rank={self.rank} world={self.world_size}")

#         train_ds, val_ds, train_loader, val_loader = self._build_loaders()
#         self._sanity_check_pasteback(train_ds if len(train_ds) > 0 else val_ds, n=8)

#         model = self._build_model().to(device)
#         if self.world_size > 1:
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)

#         optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
#         bce = nn.BCEWithLogitsLoss()
#         scaler = GradScaler("cuda", enabled=(self.amp and device.type == "cuda"))

#         best = -1.0

#         # dry-run: force 1 epoch
#         max_epochs = 1 if self.dry_run else self.epochs

#         for epoch in range(1, max_epochs + 1):
#             if self.world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
#                 train_loader.sampler.set_epoch(epoch)

#             # ---- train ----
#             model.train()
#             t0 = time.time()
#             losses = []
#             dices = []

#             for it, batch in enumerate(train_loader, start=1):
#                 img = batch["image"].to(device, non_blocking=True)
#                 msk = batch["mask"].to(device, non_blocking=True)

#                 optim.zero_grad(set_to_none=True)

#                 with autocast("cuda", enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
#                     logits = model(img)
#                     loss = 0.7 * bce(logits, msk) + 0.3 * soft_dice_loss_from_logits(logits, msk)

#                 scaler.scale(loss).backward()
#                 scaler.step(optim)
#                 scaler.update()

#                 with torch.no_grad():
#                     prob = torch.sigmoid(logits).float()
#                     d = dice_per_sample(prob, msk, thr=0.5).mean().item()

#                 losses.append(loss.item())
#                 dices.append(d)

#                 if self._is_main() and (it % self.log_freq == 0):
#                     logging.info(
#                         f"Epoch {epoch:03d} | it {it:04d}/{len(train_loader)} "
#                         f"loss {np.mean(losses):.4f} dice {np.mean(dices):.4f}"
#                     )

#                 if self.dry_run and it >= self.dry_run_train_steps:
#                     logging.info(f"[dry_run] stop train at it={it}")
#                     break

#             train_loss = float(np.mean(losses)) if losses else 0.0
#             train_dice = float(np.mean(dices)) if dices else 0.0

#             # ---- val ----
#             model.eval()
#             v_losses = []
#             dice_all = []
#             dice_nonempty = []
#             empty_gt = 0
#             empty_gt_pred_empty = 0
#             fg_fracs = []
#             pred_fracs = []

#             with torch.no_grad():
#                 for vi, batch in enumerate(val_loader, start=1):
#                     img = batch["image"].to(device, non_blocking=True)
#                     msk = batch["mask"].to(device, non_blocking=True)

#                     with autocast("cuda", enabled=(self.amp and device.type == "cuda"), dtype=torch.bfloat16):
#                         logits = model(img)
#                         loss = 0.7 * bce(logits, msk) + 0.3 * soft_dice_loss_from_logits(logits, msk)

#                     prob = torch.sigmoid(logits).float()
#                     d1 = dice_per_sample(prob, msk, thr=0.5)[0].item()
#                     v_losses.append(loss.item())
#                     dice_all.append(d1)

#                     gt_sum = float(msk.sum().item())
#                     pred_sum = float((prob > 0.5).float().sum().item())
#                     fg_fracs.append(float(msk.mean().item()))
#                     pred_fracs.append(float((prob > 0.5).float().mean().item()))

#                     if gt_sum <= 0.0:
#                         empty_gt += 1
#                         if pred_sum <= 0.0:
#                             empty_gt_pred_empty += 1
#                     else:
#                         dice_nonempty.append(d1)

#                     if self.dry_run and vi >= self.dry_run_val_steps:
#                         logging.info(f"[dry_run] stop val at vi={vi}")
#                         break

#             val_loss = float(np.mean(v_losses)) if v_losses else 0.0

#             def _stats(arr: List[float]) -> Tuple[float, float, float]:
#                 if not arr:
#                     return 0.0, 0.0, 0.0
#                 a = np.array(arr, dtype=np.float32)
#                 return float(a.mean()), float(np.quantile(a, 0.5)), float(np.quantile(a, 0.9))

#             dice_all_mean, dice_all_p50, dice_all_p90 = _stats(dice_all)
#             dice_ne_mean, dice_ne_p50, dice_ne_p90 = _stats(dice_nonempty)
#             fg_mean = float(np.mean(fg_fracs)) if fg_fracs else 0.0
#             pred_mean = float(np.mean(pred_fracs)) if pred_fracs else 0.0
#             empty_gt_rate = float(empty_gt / max(1, len(dice_all)))
#             empty_pred_rate_on_empty_gt = float(empty_gt_pred_empty / max(1, empty_gt)) if empty_gt > 0 else 0.0

#             # DDP reduce (optional)
#             if self.world_size > 1:
#                 tl = torch.tensor([train_loss, train_dice, val_loss, dice_ne_mean], device=device)
#                 torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
#                 tl = tl / float(self.world_size)
#                 train_loss, train_dice, val_loss, dice_ne_mean = [float(x) for x in tl.tolist()]

#             if self._is_main():
#                 dt = time.time() - t0
#                 logging.info(
#                     f"Epoch {epoch:03d} | train loss {train_loss:.4f} dice {train_dice:.4f} "
#                     f"| val loss {val_loss:.4f} "
#                     f"| dice_all mean {dice_all_mean:.4f} (p50 {dice_all_p50:.4f}, p90 {dice_all_p90:.4f}) "
#                     f"| dice_nonempty mean {dice_ne_mean:.4f} (p50 {dice_ne_p50:.4f}, p90 {dice_ne_p90:.4f}) "
#                     f"| empty_gt_rate {empty_gt_rate:.3f} | fg_frac mean {fg_mean:.4f} | pred_fg_frac mean {pred_mean:.4f} "
#                     f"| {dt:.1f}s"
#                 )
#                 logging.info(
#                     f"          empty_pred_rate_on_empty_gt = {empty_pred_rate_on_empty_gt:.3f} "
#                     f"(how often model predicts empty when GT is empty)"
#                 )

#             # ckpt
#             score = dice_ne_mean  # focus on nonempty
#             if score > best:
#                 best = score
#                 self._save_ckpt(model.module if hasattr(model, "module") else model, optim, epoch, best, "best.pt")
#             self._save_ckpt(model.module if hasattr(model, "module") else model, optim, epoch, best, "last.pt")

#             # debug vis
#             self._debug_crop_vis(model.module if hasattr(model, "module") else model, val_loader, epoch, device)
#             self._debug_full_vis(model.module if hasattr(model, "module") else model, val_ds, epoch, device)

#         if self._is_main():
#             logging.info(f"[DONE] best_score(dice_nonempty_mean)={best:.4f} out_dir={self.out_dir}")


# # =========================
# # Launchers (local spawn + submitit)
# # =========================
# def single_proc_run(local_rank: int, main_port: int, cfg, world_size: int):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(main_port)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     os.environ["WORLD_SIZE"] = str(world_size)

#     add_pythonpath_to_sys_path()

#     exp_dir = OmegaConf.select(cfg, "launcher.experiment_log_dir")
#     if exp_dir is None:
#         exp_dir = str(Path(os.getcwd()) / "runs" / f"unnamed_{now_ts()}")
#         OmegaConf.update(cfg, "launcher.experiment_log_dir", exp_dir, force_add=True)

#     # inject out_dir into trainer if missing
#     if OmegaConf.select(cfg, "trainer.out_dir") is None:
#         OmegaConf.update(cfg, "trainer.out_dir", exp_dir, force_add=True)

#     trainer = instantiate(cfg.trainer, _recursive_=False)
#     trainer.run()


# def single_node_runner(cfg, main_port: int):
#     num_proc = int(OmegaConf.select(cfg, "launcher.gpus_per_node") or 1)
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

#         exp_dir = OmegaConf.select(self.cfg, "launcher.experiment_log_dir")
#         if OmegaConf.select(self.cfg, "trainer.out_dir") is None:
#             OmegaConf.update(self.cfg, "trainer.out_dir", exp_dir, force_add=True)

#         trainer = instantiate(self.cfg.trainer, _recursive_=False)
#         trainer.run()


# # =========================
# # Config robustness
# # =========================
# def _ensure_cfg_defaults(cfg, args):
#     # ensure groups exist
#     if OmegaConf.select(cfg, "launcher") is None:
#         OmegaConf.update(cfg, "launcher", {}, force_add=True)
#     if OmegaConf.select(cfg, "submitit") is None:
#         OmegaConf.update(cfg, "submitit", {}, force_add=True)
#     if OmegaConf.select(cfg, "trainer") is None:
#         OmegaConf.update(cfg, "trainer", {}, force_add=True)

#     def sel(path, default=None):
#         v = OmegaConf.select(cfg, path)
#         return default if v is None else v

#     # defaults
#     OmegaConf.update(cfg, "launcher.num_nodes", int(sel("launcher.num_nodes", 1)), force_add=True)
#     OmegaConf.update(cfg, "launcher.gpus_per_node", int(sel("launcher.gpus_per_node", 1)), force_add=True)
#     OmegaConf.update(cfg, "launcher.experiment_log_dir", sel("launcher.experiment_log_dir", None), force_add=True)

#     OmegaConf.update(cfg, "submitit.use_cluster", bool(sel("submitit.use_cluster", False)), force_add=True)
#     OmegaConf.update(cfg, "submitit.timeout_hour", int(sel("submitit.timeout_hour", 24)), force_add=True)
#     OmegaConf.update(cfg, "submitit.cpus_per_task", int(sel("submitit.cpus_per_task", 8)), force_add=True)
#     OmegaConf.update(cfg, "submitit.partition", sel("submitit.partition", None), force_add=True)
#     OmegaConf.update(cfg, "submitit.name", str(sel("submitit.name", getattr(args, "config", "exp"))), force_add=True)
#     OmegaConf.update(cfg, "submitit.port_range", sel("submitit.port_range", [15000, 19999]), force_add=True)

#     # CLI overrides (struct-safe)
#     if args.num_gpus is not None:
#         OmegaConf.update(cfg, "launcher.gpus_per_node", int(args.num_gpus), force_add=True)
#     if args.num_nodes is not None:
#         OmegaConf.update(cfg, "launcher.num_nodes", int(args.num_nodes), force_add=True)
#     if args.use_cluster is not None:
#         OmegaConf.update(cfg, "submitit.use_cluster", bool(args.use_cluster), force_add=True)
#     if args.dry_run is not None:
#         OmegaConf.update(cfg, "trainer.dry_run", bool(args.dry_run), force_add=True)

#     return cfg


# def main(args):
#     cfg = compose(config_name=args.config)

#     print("###################### Loaded Config (raw) ######################")
#     print(OmegaConf.to_yaml(cfg))
#     print("#################################################################")

#     cfg = _ensure_cfg_defaults(cfg, args)

#     exp_dir = OmegaConf.select(cfg, "launcher.experiment_log_dir")
#     if exp_dir is None:
#         exp_dir = str(Path(os.getcwd()) / "runs" / f"{args.config}_{now_ts()}")
#     else:
#         exp_dir = str(Path(exp_dir) / f"{args.config}_{now_ts()}")
#     OmegaConf.update(cfg, "launcher.experiment_log_dir", exp_dir, force_add=True)
#     makedir(exp_dir)

#     print("###################### Train App Config (final) ##################")
#     print(OmegaConf.to_yaml(cfg))
#     print("#################################################################")

#     dump_cfg_files(cfg, exp_dir)
#     add_pythonpath_to_sys_path()

#     use_cluster = bool(OmegaConf.select(cfg, "submitit.use_cluster"))
#     if use_cluster:
#         submitit_dir = str(Path(exp_dir) / "submitit_logs")
#         executor = submitit.AutoExecutor(folder=submitit_dir)

#         port_range = OmegaConf.select(cfg, "submitit.port_range") or [15000, 19999]
#         port = random.randint(int(port_range[0]), int(port_range[1]))

#         job_kwargs = {
#             "timeout_min": int(OmegaConf.select(cfg, "submitit.timeout_hour")) * 60,
#             "name": str(OmegaConf.select(cfg, "submitit.name")),
#             "slurm_partition": OmegaConf.select(cfg, "submitit.partition"),
#             "gpus_per_node": int(OmegaConf.select(cfg, "launcher.gpus_per_node")),
#             "tasks_per_node": int(OmegaConf.select(cfg, "launcher.gpus_per_node")),
#             "cpus_per_task": int(OmegaConf.select(cfg, "submitit.cpus_per_task")),
#             "nodes": int(OmegaConf.select(cfg, "launcher.num_nodes")),
#         }
#         executor.update_parameters(**job_kwargs)

#         runner = SubmititRunner(port, cfg)
#         job = executor.submit(runner)
#         print("Submitit Job ID:", job.job_id)
#     else:
#         OmegaConf.update(cfg, "launcher.num_nodes", 1, force_add=True)
#         port_range = OmegaConf.select(cfg, "submitit.port_range") or [15000, 19999]
#         port = random.randint(int(port_range[0]), int(port_range[1]))
#         single_node_runner(cfg, port)


# if __name__ == "__main__":
#     # configs live at: dataflow_train/configs/<name>.yaml
#     initialize_config_module("dataflow_train.configs", version_base="1.2")

#     parser = ArgumentParser()
#     parser.add_argument("-c", "--config", required=True, type=str, help="config name under dataflow_train/configs")
#     parser.add_argument("--use-cluster", type=int, default=None, help="0 local / 1 slurm")
#     parser.add_argument("--num-gpus", type=int, default=None)
#     parser.add_argument("--num-nodes", type=int, default=None)
#     parser.add_argument("--dry-run", type=int, default=None, help="1: run quick validation (1 epoch, limited steps)")
#     args = parser.parse_args()
#     args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

#     main(args)





# # dataflow_train/train.py
# from __future__ import annotations

# import os
# import sys
# import time
# import json
# import random
# import logging
# import traceback
# from argparse import ArgumentParser
# from pathlib import Path
# from typing import Any

# import torch
# import submitit
# from hydra import compose, initialize_config_module
# from hydra.utils import instantiate
# from omegaconf import OmegaConf, open_dict


# # -------------------------
# # Make package import robust
# # -------------------------
# def ensure_repo_root_on_path():
#     this = Path(__file__).resolve()
#     repo_root = this.parents[1]  # .../dataflow (contains dataflow_train/)
#     if str(repo_root) not in sys.path:
#         sys.path.insert(0, str(repo_root))


# def add_pythonpath_to_sys_path():
#     p = os.environ.get("PYTHONPATH", "")
#     if not p:
#         return
#     for part in p.split(":"):
#         if part and part not in sys.path:
#             sys.path.insert(0, part)


# # -------------------------
# # Logging (stdout/stderr tee)
# # -------------------------
# class Tee:
#     def __init__(self, *streams):
#         self.streams = streams

#     def write(self, data):
#         for s in self.streams:
#             s.write(data)
#             s.flush()

#     def flush(self):
#         for s in self.streams:
#             s.flush()


# def setup_run_logging(out_dir: str, rank: int = 0) -> str:
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     log_path = out_dir / ("train.log" if rank == 0 else f"train_rank{rank}.log")

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[
#             logging.FileHandler(log_path, mode="a", encoding="utf-8"),
#             logging.StreamHandler(sys.__stdout__),
#         ],
#     )

#     f = open(log_path, "a", encoding="utf-8", buffering=1)
#     sys.stdout = Tee(sys.__stdout__, f)
#     sys.stderr = Tee(sys.__stderr__, f)

#     logging.info(f"[rank{rank}] Logging to: {log_path}")
#     return str(log_path)


# def format_exception(e: Exception, limit=80) -> str:
#     tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
#     return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


# def makedir(p: str):
#     Path(p).mkdir(parents=True, exist_ok=True)


# def _timestamp() -> str:
#     return time.strftime("%Y-%m-%d_%H-%M-%S")


# def _make_exp_dir(base_dir: str | None, config_name: str) -> str:
#     ts = _timestamp()
#     if base_dir is None or str(base_dir).strip() == "":
#         return str(Path(os.getcwd()) / "runs" / f"{config_name}_{ts}")

#     p = Path(base_dir)
#     # If user passes ".../runs/semantic_unet_multi" treat it as a bucket dir and create a new run under parent.
#     if p.name.endswith("_multi"):
#         return str(p.parent / f"{config_name}_{ts}")
#     # If base is an existing dir, create sub-run under it.
#     if p.exists() and p.is_dir():
#         return str(p / f"{config_name}_{ts}")
#     # Otherwise, assume it's already intended as a run dir.
#     return str(p)


# # -------------------------
# # Local/Distributed launch
# # -------------------------
# def _set_dist_env(rank: int, local_rank: int, world_size: int, master_addr: str, master_port: int):
#     os.environ["MASTER_ADDR"] = master_addr
#     os.environ["MASTER_PORT"] = str(master_port)
#     os.environ["RANK"] = str(rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     os.environ["WORLD_SIZE"] = str(world_size)


# def _worker_main(local_rank: int, master_port: int, cfg):
#     # Single-node DDP: rank == local_rank, world_size == gpus_per_node
#     world_size = int(cfg.launcher.gpus_per_node)
#     _set_dist_env(rank=local_rank, local_rank=local_rank, world_size=world_size, master_addr="127.0.0.1", master_port=master_port)

#     ensure_repo_root_on_path()
#     add_pythonpath_to_sys_path()

#     exp_dir = str(cfg.launcher.experiment_log_dir)
#     setup_run_logging(exp_dir, rank=local_rank)

#     try:
#         # instantiate trainer
#         trainer = instantiate(cfg.trainer, _recursive_=False)
#         trainer.run()
#     except Exception as e:
#         logging.error(format_exception(e))
#         raise


# def run_local(cfg):
#     assert int(cfg.launcher.num_nodes) == 1, "local runner supports single-node only"
#     gpus = int(cfg.launcher.gpus_per_node)

#     # pick a port
#     pr = cfg.submitit.port_range
#     port = random.randint(int(pr[0]), int(pr[1]))

#     # spawn
#     torch.multiprocessing.set_start_method("spawn", force=True)
#     if gpus <= 1:
#         _worker_main(local_rank=0, master_port=port, cfg=cfg)
#     else:
#         torch.multiprocessing.start_processes(
#             _worker_main,
#             args=(port, cfg),
#             nprocs=gpus,
#             start_method="spawn",
#         )


# # -------------------------
# # Submitit (optional)
# # -------------------------
# class SubmititRunner(submitit.helpers.Checkpointable):
#     def __init__(self, port: int, cfg):
#         self.cfg = cfg
#         self.port = port

#     def __call__(self):
#         ensure_repo_root_on_path()
#         add_pythonpath_to_sys_path()

#         job_env = submitit.JobEnvironment()
#         os.environ["MASTER_ADDR"] = job_env.hostnames[0]
#         os.environ["MASTER_PORT"] = str(self.port)
#         os.environ["RANK"] = str(job_env.global_rank)
#         os.environ["LOCAL_RANK"] = str(job_env.local_rank)
#         os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

#         exp_dir = str(self.cfg.launcher.experiment_log_dir)
#         setup_run_logging(exp_dir, rank=int(job_env.global_rank))

#         trainer = instantiate(self.cfg.trainer, _recursive_=False)
#         trainer.run()


# def run_submitit(cfg):
#     exp_dir = str(cfg.launcher.experiment_log_dir)
#     submitit_dir = str(Path(exp_dir) / "submitit_logs")
#     executor = submitit.AutoExecutor(folder=submitit_dir)

#     job_kwargs = {
#         "timeout_min": int(cfg.submitit.timeout_hour) * 60,
#         "name": str(cfg.submitit.name),
#         "slurm_partition": cfg.submitit.partition,
#         "gpus_per_node": int(cfg.launcher.gpus_per_node),
#         "tasks_per_node": int(cfg.launcher.gpus_per_node),
#         "cpus_per_task": int(cfg.submitit.cpus_per_task),
#         "nodes": int(cfg.launcher.num_nodes),
#     }
#     executor.update_parameters(**job_kwargs)

#     pr = cfg.submitit.port_range
#     port = random.randint(int(pr[0]), int(pr[1]))
#     runner = SubmititRunner(port, cfg)
#     job = executor.submit(runner)
#     print("Submitit Job ID:", job.job_id)


# # -------------------------
# # Main
# # -------------------------
# def main(args):
#     ensure_repo_root_on_path()
#     add_pythonpath_to_sys_path()

#     # hydra compose without @hydra.main
#     cfg = compose(config_name=args.config)

#     # allow CLI overrides without struct errors
#     with open_dict(cfg):
#         if args.num_gpus is not None:
#             cfg.launcher.gpus_per_node = int(args.num_gpus)
#         if args.num_nodes is not None:
#             cfg.launcher.num_nodes = int(args.num_nodes)
#         if args.use_cluster is not None:
#             cfg.submitit.use_cluster = bool(args.use_cluster)

#         if args.dry_run is not None:
#             cfg.trainer.dry_run = bool(args.dry_run)
#         if args.dry_run_train_steps is not None:
#             cfg.trainer.dry_run_train_steps = int(args.dry_run_train_steps)
#         if args.dry_run_val_steps is not None:
#             cfg.trainer.dry_run_val_steps = int(args.dry_run_val_steps)

#         # make a fresh run dir
#         cfg.launcher.experiment_log_dir = _make_exp_dir(cfg.launcher.experiment_log_dir, args.config)

#     exp_dir = str(cfg.launcher.experiment_log_dir)
#     makedir(exp_dir)

#     # print config (raw/final)
#     print("###################### Train App Config (final) ##################")
#     print(OmegaConf.to_yaml(cfg))
#     print("#################################################################")

#     # dump config.yaml / config_resolved.yaml
#     with open(Path(exp_dir) / "config.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg))

#     cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
#     with open(Path(exp_dir) / "config_resolved.yaml", "w", encoding="utf-8") as f:
#         f.write(OmegaConf.to_yaml(cfg_resolved))

#     # Launch
#     if bool(cfg.submitit.use_cluster):
#         run_submitit(cfg)
#     else:
#         # force local single-node
#         with open_dict(cfg):
#             cfg.launcher.num_nodes = 1
#         run_local(cfg)


# if __name__ == "__main__":
#     os.environ["HYDRA_FULL_ERROR"] = "1"

#     ensure_repo_root_on_path()
#     add_pythonpath_to_sys_path()

#     # NOTE:
#     #  - configs should be at: dataflow_train/configs/<name>.yaml
#     #  - dataflow_train/configs/__init__.py must exist
#     initialize_config_module("dataflow_train.configs", version_base="1.2")

#     parser = ArgumentParser()
#     parser.add_argument("-c", "--config", required=True, type=str, help="config name under dataflow_train/configs (no .yaml)")
#     parser.add_argument("--use-cluster", type=int, default=None, help="0 local / 1 slurm")
#     parser.add_argument("--num-gpus", type=int, default=None)
#     parser.add_argument("--num-nodes", type=int, default=None)

#     parser.add_argument("--dry-run", type=int, default=None, help="1: quick validation (limits steps)")
#     parser.add_argument("--dry-run-train-steps", type=int, default=None)
#     parser.add_argument("--dry-run-val-steps", type=int, default=None)

#     args = parser.parse_args()
#     if args.use_cluster is not None:
#         args.use_cluster = bool(args.use_cluster)
#     if args.dry_run is not None:
#         args.dry_run = bool(args.dry_run)

#     main(args)




# dataflow_train/train.py
# dataflow_train/train.py
from __future__ import annotations

import os
import sys
import time
import random
import logging
import traceback
from argparse import ArgumentParser
from pathlib import Path

import torch
import submitit
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


# -------------------------
# Make package import robust
# -------------------------
def ensure_repo_root_on_path():
    this = Path(__file__).resolve()
    repo_root = this.parents[1]  # .../dataflow (contains dataflow_train/)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def add_pythonpath_to_sys_path():
    p = os.environ.get("PYTHONPATH", "")
    if not p:
        return
    for part in p.split(":"):
        if part and part not in sys.path:
            sys.path.insert(0, part)


# -------------------------
# Logging (stdout/stderr tee)
# -------------------------
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


def format_exception(e: Exception, limit=80) -> str:
    tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


def makedir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _make_exp_dir(base_dir: str | None, config_name: str) -> str:
    ts = _timestamp()
    if base_dir is None or str(base_dir).strip() == "":
        return str(Path(os.getcwd()) / "runs" / f"{config_name}_{ts}")

    p = Path(str(base_dir))
    # "..._multi" as bucket: parent/<config>_<ts>
    if p.name.endswith("_multi"):
        return str(p.parent / f"{config_name}_{ts}")
    # if existing dir: create sub-run
    if p.exists() and p.is_dir():
        return str(p / f"{config_name}_{ts}")
    # otherwise treat as run dir
    return str(p)


# -------------------------
# Local/Distributed launch
# -------------------------
def _set_dist_env(rank: int, local_rank: int, world_size: int, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def _worker_main(local_rank: int, master_port: int, cfg):
    world_size = int(cfg.launcher.gpus_per_node)
    _set_dist_env(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        master_addr="127.0.0.1",
        master_port=master_port,
    )

    ensure_repo_root_on_path()
    add_pythonpath_to_sys_path()

    exp_dir = str(cfg.launcher.experiment_log_dir)
    setup_run_logging(exp_dir, rank=local_rank)

    try:
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
    except Exception as e:
        logging.error(format_exception(e))
        raise


def run_local(cfg):
    assert int(cfg.launcher.num_nodes) == 1, "local runner supports single-node only"
    gpus = int(cfg.launcher.gpus_per_node)

    pr = cfg.submitit.port_range
    port = random.randint(int(pr[0]), int(pr[1]))

    torch.multiprocessing.set_start_method("spawn", force=True)
    if gpus <= 1:
        _worker_main(local_rank=0, master_port=port, cfg=cfg)
    else:
        torch.multiprocessing.start_processes(
            _worker_main,
            args=(port, cfg),
            nprocs=gpus,
            start_method="spawn",
        )


# -------------------------
# Submitit (optional)
# -------------------------
class SubmititRunner(submitit.helpers.Checkpointable):
    def __init__(self, port: int, cfg):
        self.cfg = cfg
        self.port = port

    def __call__(self):
        ensure_repo_root_on_path()
        add_pythonpath_to_sys_path()

        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        exp_dir = str(self.cfg.launcher.experiment_log_dir)
        setup_run_logging(exp_dir, rank=int(job_env.global_rank))

        trainer = instantiate(self.cfg.trainer, _recursive_=False)
        trainer.run()


def run_submitit(cfg):
    exp_dir = str(cfg.launcher.experiment_log_dir)
    submitit_dir = str(Path(exp_dir) / "submitit_logs")
    executor = submitit.AutoExecutor(folder=submitit_dir)

    job_kwargs = {
        "timeout_min": int(cfg.submitit.timeout_hour) * 60,
        "name": str(cfg.submitit.name),
        "slurm_partition": cfg.submitit.partition,
        "gpus_per_node": int(cfg.launcher.gpus_per_node),
        "tasks_per_node": int(cfg.launcher.gpus_per_node),
        "cpus_per_task": int(cfg.submitit.cpus_per_task),
        "nodes": int(cfg.launcher.num_nodes),
    }
    executor.update_parameters(**job_kwargs)

    pr = cfg.submitit.port_range
    port = random.randint(int(pr[0]), int(pr[1]))
    runner = SubmititRunner(port, cfg)
    job = executor.submit(runner)
    print("Submitit Job ID:", job.job_id)


def _ensure_default_sections(cfg, config_name: str):
    """
    只补齐 launcher/submitit/trainer 这三个 section 本身，不再给 trainer 注入 dry_run 字段。
    否则会把 dry_run 传给所有 trainer（比如 SAM3ExternalTrainer）导致 init 参数不匹配。
    """
    with open_dict(cfg):
        if "launcher" not in cfg:
            cfg.launcher = {}
        cfg.launcher.setdefault("num_nodes", 1)
        cfg.launcher.setdefault("gpus_per_node", 1)
        cfg.launcher.setdefault("experiment_log_dir", "")

        if "submitit" not in cfg:
            cfg.submitit = {}
        cfg.submitit.setdefault("use_cluster", False)
        cfg.submitit.setdefault("timeout_hour", 24)
        cfg.submitit.setdefault("cpus_per_task", 8)
        cfg.submitit.setdefault("partition", None)
        cfg.submitit.setdefault("name", config_name)
        cfg.submitit.setdefault("port_range", [15000, 19999])

        if "trainer" not in cfg:
            cfg.trainer = {}


def main(args):
    ensure_repo_root_on_path()
    add_pythonpath_to_sys_path()

    cfg = compose(config_name=args.config)
    _ensure_default_sections(cfg, args.config)

    # CLI overrides
    with open_dict(cfg):
        if args.num_gpus is not None:
            cfg.launcher.gpus_per_node = int(args.num_gpus)
        if args.num_nodes is not None:
            cfg.launcher.num_nodes = int(args.num_nodes)
        if args.use_cluster is not None:
            cfg.submitit.use_cluster = bool(args.use_cluster)

        # 关键：dry-run 只在 trainer 本身定义了该字段时才覆盖（避免传给不支持的 trainer）
        if args.dry_run is not None:
            if "dry_run" in cfg.trainer:
                cfg.trainer.dry_run = bool(args.dry_run)
            else:
                print("[warn] --dry-run provided but cfg.trainer has no 'dry_run'. Ignored for this trainer.")

        if args.dry_run_train_steps is not None:
            if "dry_run_train_steps" in cfg.trainer:
                cfg.trainer.dry_run_train_steps = int(args.dry_run_train_steps)
            else:
                print("[warn] --dry-run-train-steps provided but cfg.trainer has no 'dry_run_train_steps'. Ignored.")

        if args.dry_run_val_steps is not None:
            if "dry_run_val_steps" in cfg.trainer:
                cfg.trainer.dry_run_val_steps = int(args.dry_run_val_steps)
            else:
                print("[warn] --dry-run-val-steps provided but cfg.trainer has no 'dry_run_val_steps'. Ignored.")

        # create a fresh run dir
        cfg.launcher.experiment_log_dir = _make_exp_dir(cfg.launcher.experiment_log_dir, args.config)

        # if trainer has out_dir but user forgot to set, fill it
        if "out_dir" in cfg.trainer and (cfg.trainer.out_dir is None or str(cfg.trainer.out_dir).strip() == ""):
            cfg.trainer.out_dir = cfg.launcher.experiment_log_dir

    exp_dir = str(cfg.launcher.experiment_log_dir)
    makedir(exp_dir)

    # print config
    print("###################### Train App Config (final) ##################")
    print(OmegaConf.to_yaml(cfg))
    print("#################################################################")

    # dump configs
    with open(Path(exp_dir) / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open(Path(exp_dir) / "config_resolved.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg_resolved))

    # launch
    if bool(cfg.submitit.use_cluster):
        run_submitit(cfg)
    else:
        with open_dict(cfg):
            cfg.launcher.num_nodes = 1
        run_local(cfg)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    ensure_repo_root_on_path()
    add_pythonpath_to_sys_path()

    initialize_config_module("dataflow_train.configs", version_base="1.2")

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="config name under dataflow_train/configs (no .yaml)")
    parser.add_argument("--use-cluster", type=int, default=None, help="0 local / 1 slurm")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)

    parser.add_argument("--dry-run", type=int, default=None, help="1: quick validation (only if trainer supports)")
    parser.add_argument("--dry-run-train-steps", type=int, default=None)
    parser.add_argument("--dry-run-val-steps", type=int, default=None)

    args = parser.parse_args()
    if args.use_cluster is not None:
        args.use_cluster = bool(args.use_cluster)
    if args.dry_run is not None:
        args.dry_run = bool(args.dry_run)

    main(args)