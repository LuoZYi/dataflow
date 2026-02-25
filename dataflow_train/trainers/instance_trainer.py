# dataflow_train/trainers/instance_trainer.py
from __future__ import annotations

import os
import time
import random
import logging
import contextlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataflow_train.data.instance_dataset import ParquetInstanceCropDataset
from dataflow_train.models.hovernet import HoverNetLite
from dataflow_train.utils.metrics import dice_per_sample, stats
from dataflow_train.utils.vis_instance import save_instance_side_by_side


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


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _grad_scaler(device: torch.device, enabled: bool):
    use = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(enabled=use)
        except TypeError:
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=use)

    class _Dummy:
        def scale(self, loss): return loss
        def step(self, optim): optim.step()
        def update(self): pass
    return _Dummy()


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/target: [B,2,H,W], mask: [B,1,H,W] {0,1}
    """
    mask2 = mask.repeat(1, 2, 1, 1)
    diff = F.smooth_l1_loss(pred * mask2, target * mask2, reduction="sum")
    denom = mask2.sum().clamp_min(eps)
    return diff / denom


class HoverNetTrainer:
    """
    MVP instance trainer:
      - NP head: nuclei pixel (BCE)
      - HV head: regression (SmoothL1) masked by nuclei
      - Metrics: nuclei Dice only (for now)
      - Debug: save worst patches with inst_map + nuclei gt/pred overlays
    """

    def __init__(
        self,
        out_dir: str,
        db_root: str,
        dataset_roots: Dict[str, str],
        datasets: List[str],
        ann_file: str = "ann_instance.parquet",
        patch_size: int = 512,
        epochs: int = 30,
        train_epoch_size: int = 4000,
        batch_size: int = 6,
        num_workers: int = 4,
        lr: float = 3e-4,
        wd: float = 1e-4,
        seed: int = 42,
        amp: bool = True,
        log_freq: int = 20,
        use_meta_split: bool = True,
        val_ratio: float = 0.1,
        pos_fraction: float = 0.8,
        cache_slides: int = 64,
        np_loss_w: float = 1.0,
        hv_loss_w: float = 2.0,
        debug_vis_patches: int = 20,
        debug_vis_thr: float = 0.5,
        debug_vis_max_side: int = 1600,
        dry_run: bool = False,
        dry_run_train_steps: int = 50,
        dry_run_val_steps: int = 50,
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
        self.np_loss_w = float(np_loss_w)
        self.hv_loss_w = float(hv_loss_w)
        self.debug_vis_patches = int(debug_vis_patches)
        self.debug_vis_thr = float(debug_vis_thr)
        self.debug_vis_max_side = int(debug_vis_max_side)
        self.dry_run = bool(dry_run)
        self.dry_run_train_steps = int(dry_run_train_steps)
        self.dry_run_val_steps = int(dry_run_val_steps)

        self.rank = _rank()
        self.world_size = _world()
        self.local_rank = _local_rank()

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
        nw = 0 if self.dry_run else self.num_workers

        train_ds = ParquetInstanceCropDataset(
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
        val_ds = ParquetInstanceCropDataset(
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
        return HoverNetLite(in_ch=3, base=32)

    def _save_ckpt(self, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, best: float, name: str):
        if not _is_main():
            return
        ckpt_dir = Path(self.out_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        obj = {"epoch": int(epoch), "best_score": float(best), "model": model.state_dict(), "optim": optim.state_dict()}
        torch.save(obj, str(ckpt_dir / name))

    def run(self):
        self._init_dist()
        device = self._device()

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

        logging.info(f"device={device} rank={self.rank} world={self.world_size}")

        train_ds, val_ds, train_loader, val_loader = self._build_loaders()

        model = self._build_model().to(device)
        if self.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)

        optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        bce = nn.BCEWithLogitsLoss()
        scaler = _grad_scaler(device, enabled=self.amp)

        best = -1.0
        max_epochs = 1 if self.dry_run else self.epochs

        for epoch in range(1, max_epochs + 1):
            if self.world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # ---------------- train ----------------
            model.train()
            t0 = time.time()
            tr_losses, tr_dices = [], []

            for it, batch in enumerate(train_loader, start=1):
                img = batch["image"].to(device, non_blocking=True)          # [B,3,H,W]
                nuc = batch["nuclei"].to(device, non_blocking=True)         # [B,1,H,W]
                hv_gt = batch["hv"].to(device, non_blocking=True)           # [B,2,H,W]

                optim.zero_grad(set_to_none=True)

                with _autocast_ctx(device, enabled=self.amp):
                    out = model(img)
                    np_logits = out["np_logits"]
                    hv_pred = out["hv"]

                    loss_np = bce(np_logits, nuc)
                    loss_hv = _masked_smooth_l1(hv_pred, hv_gt, nuc)
                    loss = self.np_loss_w * loss_np + self.hv_loss_w * loss_hv

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                with torch.no_grad():
                    prob = torch.sigmoid(np_logits).float()
                    d = dice_per_sample(prob, nuc, thr=0.5).mean().item()

                tr_losses.append(float(loss.item()))
                tr_dices.append(float(d))

                if _is_main() and it % self.log_freq == 0:
                    logging.info(
                        f"Epoch {epoch:03d} | it {it:04d}/{len(train_loader)} "
                        f"loss {np.mean(tr_losses):.4f} dice_np {np.mean(tr_dices):.4f}"
                    )

                if self.dry_run and it >= self.dry_run_train_steps:
                    logging.info(f"[dry_run] stop train at it={it}")
                    break

            train_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
            train_dice = float(np.mean(tr_dices)) if tr_dices else 0.0

            # ---------------- val ----------------
            model.eval()
            v_losses, dice_all, dice_nonempty = [], [], []
            worst_bank = []  # store (dice, img_u8, inst_map, nuc_gt, prob, meta)

            with torch.no_grad():
                for vi, batch in enumerate(val_loader, start=1):
                    img = batch["image"].to(device, non_blocking=True)
                    nuc = batch["nuclei"].to(device, non_blocking=True)
                    hv_gt = batch["hv"].to(device, non_blocking=True)

                    with _autocast_ctx(device, enabled=self.amp):
                        out = model(img)
                        np_logits = out["np_logits"]
                        hv_pred = out["hv"]
                        loss_np = bce(np_logits, nuc)
                        loss_hv = _masked_smooth_l1(hv_pred, hv_gt, nuc)
                        loss = self.np_loss_w * loss_np + self.hv_loss_w * loss_hv

                    prob = torch.sigmoid(np_logits).float()
                    d = float(dice_per_sample(prob, nuc, thr=0.5)[0].item())

                    v_losses.append(float(loss.item()))
                    dice_all.append(d)
                    if float(nuc.sum().item()) > 0:
                        dice_nonempty.append(d)

                    if _is_main() and self.debug_vis_patches > 0:
                        img_u8 = (batch["image"][0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        inst_map = batch["inst_map"][0].detach().cpu().numpy().astype(np.int32)
                        nuc_gt = (batch["nuclei"][0, 0].detach().cpu().numpy() > 0.5)
                        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
                        suid = batch["slide_uid"][0]
                        cx, cy = batch["crop_xy"][0].tolist() if torch.is_tensor(batch["crop_xy"]) else batch["crop_xy"][0]
                        worst_bank.append((d, img_u8, inst_map, nuc_gt, prob_np, str(suid), int(cx), int(cy), batch.get("dataset", [""])[0]))

                    if self.dry_run and vi >= self.dry_run_val_steps:
                        break

            if _is_dist():
                # simple: average scalars across ranks (lists not gathered here for MVP)
                tl = torch.tensor([train_loss, train_dice, float(np.mean(v_losses)) if v_losses else 0.0], device=device)
                torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
                tl = tl / float(self.world_size)
                train_loss, train_dice, val_loss = [float(x) for x in tl.tolist()]
            else:
                val_loss = float(np.mean(v_losses)) if v_losses else 0.0

            if _is_main():
                d_all_mean, d_all_p50, d_all_p90 = stats(dice_all)
                d_ne_mean, d_ne_p50, d_ne_p90 = stats(dice_nonempty)
                dt = time.time() - t0
                logging.info(
                    f"Epoch {epoch:03d} | train loss {train_loss:.4f} dice_np {train_dice:.4f} "
                    f"| val loss {val_loss:.4f} | "
                    f"dice_all mean {d_all_mean:.4f} (p50 {d_all_p50:.4f}, p90 {d_all_p90:.4f}) | "
                    f"dice_nonempty mean {d_ne_mean:.4f} (p50 {d_ne_p50:.4f}, p90 {d_ne_p90:.4f}) | {dt:.1f}s"
                )

                core = model.module if hasattr(model, "module") else model
                score = d_ne_mean
                if score > best:
                    best = score
                    self._save_ckpt(core, optim, epoch, best, "best.pt")
                self._save_ckpt(core, optim, epoch, best, "last.pt")

                # save worst patches
                if self.debug_vis_patches > 0 and len(worst_bank) > 0:
                    worst_bank.sort(key=lambda x: x[0])  # by dice asc
                    pick = worst_bank[: min(self.debug_vis_patches, len(worst_bank))]
                    out_dir = Path(self.out_dir) / "debug_vis_patches" / f"epoch{epoch:03d}" / "worst"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    for i, (d, img_u8, inst_map, nuc_gt, prob_np, suid, cx, cy, dsname) in enumerate(pick):
                        name = f"{i:02d}_dice{d:.3f}_{dsname}_{_sanitize(suid)}_x{cx}_y{cy}.png"
                        text = f"dice={d:.3f} thr={self.debug_vis_thr:.2f} {dsname} {suid} ({cx},{cy}) inst={int(inst_map.max())}"
                        save_instance_side_by_side(
                            img_u8=img_u8,
                            inst_map=inst_map,
                            nuclei_gt=nuc_gt,
                            nuclei_prob=prob_np,
                            out_path=str(out_dir / name),
                            thr=self.debug_vis_thr,
                            max_side=self.debug_vis_max_side,
                            text=text,
                        )
                    logging.info(f"[debug_patches] saved worst-{len(pick)} to: {out_dir}")

        if _is_main():
            logging.info(f"[DONE] best_score(dice_nonempty_mean)={best:.4f} out_dir={self.out_dir}")