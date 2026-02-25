# dataflow_train/trainers/semantic_trainer.py
from __future__ import annotations

import os
import time
import math
import random
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataflow_train.data.semantic_dataset import ParquetSemanticCropDataset
from dataflow_train.models.unet import UNetSmall
from dataflow_train.utils.metrics import dice_per_sample, stats
from dataflow_train.utils.vis_semantic import save_side_by_side
from dataflow_train.utils.val_inspector import ValInspector



def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(":", "_").replace("\\", "_")


def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


# def _autocast_ctx(device: torch.device, enabled: bool):
#     # torch.cuda.amp.* is deprecated in newer torch; torch.amp.* is preferred.
#     if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
#         return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")
#     # fallback
#     return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda", dtype=torch.bfloat16)


# def _grad_scaler(device: torch.device, enabled: bool):
#     if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
#         return torch.amp.GradScaler(device_type="cuda", enabled=enabled and device.type == "cuda")
#     return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")

# def _autocast_ctx(device: torch.device, enabled: bool):
#     """
#     Most compatible autocast:
#       - CUDA: torch.cuda.amp.autocast
#       - CPU: disabled (bf16 autocast on cpu is optional and often slow/buggy)
#     """
#     if not enabled:
#         return torch.cuda.amp.autocast(enabled=False)

#     if device.type == "cuda":
#         # bfloat16 on cuda is fine on Ampere+/Hopper; if your GPU doesn't support bf16 well,
#         # switch dtype to torch.float16.
#         return torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)

#     # CPU path: disable autocast for safety
#     return torch.cuda.amp.autocast(enabled=False)


# def _grad_scaler(device: torch.device, enabled: bool):
#     """
#     Most compatible GradScaler:
#       - CUDA: torch.cuda.amp.GradScaler
#       - CPU: disabled
#     """
#     if device.type == "cuda" and enabled:
#         return torch.cuda.amp.GradScaler(enabled=True)
#     return torch.cuda.amp.GradScaler(enabled=False)


import inspect
import contextlib

def _autocast_ctx(device: torch.device, enabled: bool):
    """
    Prefer torch.amp.autocast (new API) to avoid FutureWarning.
    Fallback to torch.cuda.amp.autocast on older versions.
    """
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()

    # Newer torch: torch.amp.autocast("cuda", dtype=..., enabled=...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True)
        except TypeError:
            # some versions use device_type kw
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    # Fallback (older): torch.cuda.amp.autocast
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)

    return contextlib.nullcontext()


def _grad_scaler(device: torch.device, enabled: bool):
    """
    Prefer torch.amp.GradScaler if available and compatible.
    Fallback to torch.cuda.amp.GradScaler.
    """
    use = bool(enabled and device.type == "cuda")

    # Newer torch: torch.amp.GradScaler(...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            # some versions accept (enabled=...)
            return torch.amp.GradScaler(enabled=use)
        except TypeError:
            # fallback to cuda amp
            pass

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=use)

    # no scaler available
    class _DummyScaler:
        def scale(self, loss): return loss
        def step(self, optim): optim.step()
        def update(self): pass
    return _DummyScaler()




def build_full_gt_mask(ds: ParquetSemanticCropDataset, slide_uid: str, H: int, W: int) -> np.ndarray:
    """
    Full GT mask by pasting all decoded ROI masks onto (H,W).
    Uses ds._get_slide_decoded_rois.
    """
    full = np.zeros((H, W), dtype=bool)
    rois = ds._get_slide_decoded_rois(slide_uid)  # [(rx,ry,roi_mask,w,h)]
    for (rx, ry, roi_mask, mw, mh) in rois:
        y0, x0 = int(ry), int(rx)
        y1 = min(H, y0 + roi_mask.shape[0])
        x1 = min(W, x0 + roi_mask.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        sub = roi_mask[: (y1 - y0), : (x1 - x0)]
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
    Overlap-tile inference to get full prob map [H,W].
    """
    H, W = img_u8.shape[:2]
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)

    model.eval()

    ys = list(range(0, max(1, H - patch + 1), stride))
    xs = list(range(0, max(1, W - patch + 1), stride))
    if ys[-1] != max(0, H - patch):
        ys.append(max(0, H - patch))
    if xs[-1] != max(0, W - patch):
        xs.append(max(0, W - patch))

    for y in ys:
        for x in xs:
            tile = img_u8[y:y + patch, x:x + patch]
            if tile.shape[0] != patch or tile.shape[1] != patch:
                pad = np.zeros((patch, patch, 3), dtype=np.uint8)
                pad[: tile.shape[0], : tile.shape[1]] = tile
                tile = pad

            t = torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device, non_blocking=True)

            with _autocast_ctx(device, enabled=amp):
                logits = model(t)
                prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

            yh = min(H, y + patch)
            xw = min(W, x + patch)
            ph = yh - y
            pw = xw - x
            pred_sum[y:yh, x:xw] += prob[:ph, :pw]
            pred_cnt[y:yh, x:xw] += 1.0

    return pred_sum / np.maximum(pred_cnt, 1e-6)


class SemanticUNetTrainer:
    """
    Hydra instantiate target:
      trainer:
        _target_: dataflow_train.trainers.semantic_trainer.SemanticUNetTrainer
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
        # validation / split
        use_meta_split: bool = True,
        val_ratio: float = 0.1,
        # sampling/cache
        pos_fraction: float = 0.7,
        cache_slides: int = 64,
        # debug vis
        debug_vis_full: int = 4,
        debug_vis_thr: float = 0.5,
        debug_vis_max_side: int = 1600,
        stride: int = 256,
        # extra debug
        sanity_check_n: int = 8,
        # dry run
        dry_run: bool = False,
        dry_run_train_steps: int = 50,
        dry_run_val_steps: int = 20,
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

        self.use_meta_split = bool(use_meta_split)
        self.val_ratio = float(val_ratio)

        self.pos_fraction = float(pos_fraction)
        self.cache_slides = int(cache_slides)

        self.debug_vis_full = int(debug_vis_full)
        self.debug_vis_thr = float(debug_vis_thr)
        self.debug_vis_max_side = int(debug_vis_max_side)
        self.stride = int(stride)

        self.sanity_check_n = int(sanity_check_n)

        self.dry_run = bool(dry_run)
        self.dry_run_train_steps = int(dry_run_train_steps)
        self.dry_run_val_steps = int(dry_run_val_steps)

        # runtime env
        self.rank = _rank()
        self.world_size = _world()
        self.local_rank = _local_rank()

        # seed per rank
        _seed_all(self.seed + self.rank)

    def _init_dist(self):
        if self.world_size <= 1:
            return
        if torch.distributed.is_initialized():
            return
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(self.local_rank)

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def _build_loaders(self):
        # IMPORTANT: dry-run 强制 num_workers=0，避免你之前看到的 DataLoader worker abort（break early + worker teardown）
        nw = 0 if self.dry_run else self.num_workers

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
            use_meta_split=self.use_meta_split,
            val_ratio=self.val_ratio,
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
            use_meta_split=self.use_meta_split,
            val_ratio=self.val_ratio,
        )

        train_sampler = None
        val_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=(not self.dry_run),
            drop_last=True,
            persistent_workers=(nw > 0 and (not self.dry_run)),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=0 if self.dry_run else max(1, nw // 2),
            pin_memory=(not self.dry_run),
            drop_last=False,
        )
        return train_ds, val_ds, train_loader, val_loader

    def _build_model(self) -> nn.Module:
        return UNetSmall(in_ch=3, out_ch=1, base=32)

    def _save_ckpt(self, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, best_score: float, name: str):
        if not _is_main():
            return
        ckpt_dir = Path(self.out_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        obj = {
            "epoch": int(epoch),
            "best_score": float(best_score),
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        }
        torch.save(obj, str(path))

    def _sanity_check_pasteback(self, ds: ParquetSemanticCropDataset):
        if not _is_main():
            return
        n = max(0, self.sanity_check_n)
        if n <= 0:
            return
        if len(ds.slide_uids) == 0:
            return

        ps = self.patch_size
        ok = 0
        for _ in range(n):
            suid = random.choice(ds.slide_uids)
            mr = ds.meta[ds.meta["slide_uid"].astype(str) == suid].iloc[0].to_dict()
            H, W = int(mr["height_px"]), int(mr["width_px"])

            full = build_full_gt_mask(ds, suid, H=H, W=W)
            if H <= ps or W <= ps:
                cx, cy = 0, 0
            else:
                cx = random.randint(0, W - ps)
                cy = random.randint(0, H - ps)

            # dataset crop mask
            try:
                crop_mask = ds._make_crop_mask(suid, cx, cy, ps).astype(bool)
            except Exception:
                # 如果你把 dataset 的内部方法名改了，就跳过 sanity（不影响训练）
                logging.warning("[sanity] ds._make_crop_mask not accessible; skip sanity check.")
                return

            ref = full[cy:cy + ps, cx:cx + ps].astype(bool)
            if crop_mask.shape != ref.shape:
                raise RuntimeError(f"[sanity] shape mismatch {crop_mask.shape} vs {ref.shape}")

            if not np.array_equal(crop_mask, ref):
                diff = np.logical_xor(crop_mask, ref).mean()
                raise RuntimeError(f"[sanity] pasteback mismatch! diff_rate={diff:.6f} slide={suid} crop=({cx},{cy})")
            ok += 1

        logging.info(f"[sanity] pasteback OK: {ok} samples, no mismatch")

    def _debug_full_vis(self, model: nn.Module, val_ds: ParquetSemanticCropDataset, epoch: int, device: torch.device):
        if not _is_main():
            return
        if self.debug_vis_full <= 0:
            return
        if len(val_ds.slide_uids) == 0:
            logging.info("[debug_full] no val slides")
            return

        out_dir = Path(self.out_dir) / "debug_vis_full" / f"epoch{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pick = val_ds.slide_uids[: min(self.debug_vis_full, len(val_ds.slide_uids))]
        for suid in pick:
            mr = val_ds.meta[val_ds.meta["slide_uid"].astype(str) == suid].iloc[0].to_dict()
            # load full image through dataset helper
            img_full = val_ds._load_image_full(mr)  # uint8
            H, W = img_full.shape[:2]
            gt_full = build_full_gt_mask(val_ds, suid, H=H, W=W)

            pred_full = tiled_predict_full(
                model=model,
                img_u8=img_full,
                device=device,
                patch=self.patch_size,
                stride=self.stride,
                amp=self.amp,
            )

            out_path = str(out_dir / f"{_sanitize(suid)}.png")
            save_side_by_side(
                img_u8=img_full,
                gt=gt_full,
                pred_prob=pred_full,
                out_path=out_path,
                thr=self.debug_vis_thr,
                max_side=self.debug_vis_max_side,
            )
        logging.info(f"[debug_full] saved to: {out_dir}")

    def run(self):
        self._init_dist()

        device = self._device()
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        logging.info(f"device={device} rank={self.rank} world={self.world_size}")
        logging.info(f"trainer_cfg: {asdict(self) if False else '(see config dump)'}")

        train_ds, val_ds, train_loader, val_loader = self._build_loaders()

        # sanity check pasteback (rank0 only)
        self._sanity_check_pasteback(train_ds)

        model = self._build_model().to(device)
        if self.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )

        optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        bce = nn.BCEWithLogitsLoss()
        scaler = _grad_scaler(device, enabled=self.amp)

        best = -1.0

        max_epochs = 1 if self.dry_run else self.epochs
        for epoch in range(1, max_epochs + 1):
            if self.world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # -------- train --------
            model.train()
            t0 = time.time()
            losses: List[float] = []
            dices: List[float] = []

            for it, batch in enumerate(train_loader, start=1):
                img = batch["image"].to(device, non_blocking=True)
                msk = batch["mask"].to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)

                with _autocast_ctx(device, enabled=self.amp):
                    logits = model(img)
                    loss = 0.7 * bce(logits, msk) + 0.3 * soft_dice_loss_from_logits(logits, msk)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                with torch.no_grad():
                    prob = torch.sigmoid(logits)
                    d = dice_per_sample(prob, msk, thr=0.5).mean().item()

                losses.append(float(loss.item()))
                dices.append(float(d))

                if _is_main() and (it % self.log_freq == 0):
                    logging.info(
                        f"Epoch {epoch:03d} | it {it:04d}/{len(train_loader)} "
                        f"loss {np.mean(losses):.4f} dice {np.mean(dices):.4f}"
                    )

                if self.dry_run and it >= self.dry_run_train_steps:
                    logging.info(f"[dry_run] stop train at it={it}")
                    break

            train_loss = float(np.mean(losses)) if losses else 0.0
            train_dice = float(np.mean(dices)) if dices else 0.0

            # -------- val --------
            model.eval()
            v_losses: List[float] = []
            dice_all: List[float] = []
            dice_nonempty: List[float] = []
            empty_gt = 0
            empty_pred_on_empty_gt = 0
            fg_fracs: List[float] = []
            pred_fg_fracs: List[float] = []
            # ---- val inspector: worst patches + thr sweep (rank0 only) ----
            insp = None
            if _is_main():
                insp = ValInspector(
                    out_dir=self.out_dir,
                    epoch=epoch,
                    k=20,
                    thrs=[0.3, 0.5, 0.7],
                    save_thr=float(self.debug_vis_thr),
                    max_side=int(self.debug_vis_max_side),
                )
            with torch.no_grad():
                for vi, batch in enumerate(val_loader, start=1):
                    img = batch["image"].to(device, non_blocking=True)
                    msk = batch["mask"].to(device, non_blocking=True)

                    with _autocast_ctx(device, enabled=self.amp):
                        logits = model(img)
                        loss = 0.7 * bce(logits, msk) + 0.3 * soft_dice_loss_from_logits(logits, msk)

                    prob = torch.sigmoid(logits).float()
                    d = dice_per_sample(prob, msk, thr=0.5)[0].item()
                    v_losses.append(float(loss.item()))
                    dice_all.append(float(d))
                    if insp is not None:
                        prob_np = prob[0, 0].detach().cpu().numpy()
                        gt_np = (msk[0, 0].detach().cpu().numpy() > 0.5)
                        insp.update(batch, prob_np, gt_np, dice=float(d))


                    gt_sum = float(msk.sum().item())
                    fg_fracs.append(float(msk.mean().item()))
                    pred_bin = (prob > 0.5).float()
                    pred_fg_fracs.append(float(pred_bin.mean().item()))

                    if gt_sum <= 0.0:
                        empty_gt += 1
                        if float(pred_bin.sum().item()) <= 0.0:
                            empty_pred_on_empty_gt += 1
                    else:
                        dice_nonempty.append(float(d))

                    if self.dry_run and vi >= self.dry_run_val_steps:
                        logging.info(f"[dry_run] stop val at vi={vi}")
                        break

            # --- gather val stats in DDP ---
            if _is_dist():
                # gather python lists to rank0 for correct quantiles
                obj = {
                    "v_losses": v_losses,
                    "dice_all": dice_all,
                    "dice_nonempty": dice_nonempty,
                    "empty_gt": empty_gt,
                    "empty_pred_on_empty_gt": empty_pred_on_empty_gt,
                    "fg_fracs": fg_fracs,
                    "pred_fg_fracs": pred_fg_fracs,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                }
                gathered = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(gathered, obj)

                if _is_main():
                    v_losses = []
                    dice_all = []
                    dice_nonempty = []
                    fg_fracs = []
                    pred_fg_fracs = []
                    empty_gt = 0
                    empty_pred_on_empty_gt = 0
                    train_loss_list = []
                    train_dice_list = []
                    for g in gathered:
                        v_losses += g["v_losses"]
                        dice_all += g["dice_all"]
                        dice_nonempty += g["dice_nonempty"]
                        fg_fracs += g["fg_fracs"]
                        pred_fg_fracs += g["pred_fg_fracs"]
                        empty_gt += int(g["empty_gt"])
                        empty_pred_on_empty_gt += int(g["empty_pred_on_empty_gt"])
                        train_loss_list.append(float(g["train_loss"]))
                        train_dice_list.append(float(g["train_dice"]))
                    train_loss = float(np.mean(train_loss_list)) if train_loss_list else train_loss
                    train_dice = float(np.mean(train_dice_list)) if train_dice_list else train_dice

            # rank0 logs/ckpt/vis
            if _is_main():
                val_loss = float(np.mean(v_losses)) if v_losses else 0.0

                d_all_mean, d_all_p50, d_all_p90 = stats(dice_all)
                d_ne_mean, d_ne_p50, d_ne_p90 = stats(dice_nonempty)

                n_all = max(1, len(dice_all))
                empty_gt_rate = float(empty_gt) / float(n_all)
                empty_pred_rate = float(empty_pred_on_empty_gt) / float(max(1, empty_gt))

                fg_mean = float(np.mean(fg_fracs)) if fg_fracs else 0.0
                pred_fg_mean = float(np.mean(pred_fg_fracs)) if pred_fg_fracs else 0.0

                dt = time.time() - t0
                logging.info(
                    f"Epoch {epoch:03d} | train loss {train_loss:.4f} dice {train_dice:.4f} "
                    f"| val loss {val_loss:.4f} | "
                    f"dice_all mean {d_all_mean:.4f} (p50 {d_all_p50:.4f}, p90 {d_all_p90:.4f}) | "
                    f"dice_nonempty mean {d_ne_mean:.4f} (p50 {d_ne_p50:.4f}, p90 {d_ne_p90:.4f}) | "
                    f"empty_gt_rate {empty_gt_rate:.3f} | fg_frac mean {fg_mean:.4f} | {dt:.1f}s"
                )
                logging.info(
                    f"          empty_pred_rate_on_empty_gt = {empty_pred_rate:.3f} "
                    f"(how often model predicts empty when GT is empty) | pred_fg_frac mean {pred_fg_mean:.4f}"
                )
                


                # score for "best"
                score = d_ne_mean  # non-empty dice mean is usually more meaningful
                core_model = model.module if hasattr(model, "module") else model
                if score > best:
                    best = score
                    self._save_ckpt(core_model, optim, epoch, best, "best.pt")
                self._save_ckpt(core_model, optim, epoch, best, "last.pt")

                # debug full vis
                self._debug_full_vis(core_model, val_ds, epoch, device)

        if _is_main():
            logging.info(f"[DONE] best_score(dice_nonempty_mean)={best:.4f} out_dir={self.out_dir}")

        # help GC to shutdown dataloader workers cleanly (esp. if you later enable workers)
        try:
            del train_loader, val_loader
        except Exception:
            pass
