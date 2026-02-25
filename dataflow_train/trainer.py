# dataflow_train/trainer.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .data_semantic import ParquetSemanticDataset
from .models_unet import UNet
from .losses import loss_bce_dice, dice_score, save_debug_overlays


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SemanticUNetTrainer:
    experiment_log_dir: str
    db_root: str
    dataset_roots: Dict[str, str]
    datasets: List[str]
    ann_name: str
    patch: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    amp: bool
    device: str
    seed: int = 0
    debug_vis: int = 0

    def run(self):
        seed_all(int(self.seed))
        exp = Path(self.experiment_log_dir)
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "ckpt").mkdir(exist_ok=True)
        (exp / "debug_vis").mkdir(exist_ok=True)

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        train_sets = []
        val_sets = []
        for ds in self.datasets:
            train_sets.append(
                ParquetSemanticDataset(
                    db_root=Path(self.db_root),
                    ds_name=ds,
                    ds_root=Path(self.dataset_roots[ds]),
                    ann_name=self.ann_name,
                    split="train",
                    patch=int(self.patch),
                    train_mode=True,
                    seed=int(self.seed),
                )
            )
            val_sets.append(
                ParquetSemanticDataset(
                    db_root=Path(self.db_root),
                    ds_name=ds,
                    ds_root=Path(self.dataset_roots[ds]),
                    ann_name=self.ann_name,
                    split="val",
                    patch=int(self.patch),
                    train_mode=False,
                    seed=int(self.seed),
                )
            )

        train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
        val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

        train_loader = DataLoader(
            train_ds, batch_size=int(self.batch_size), shuffle=True,
            num_workers=int(self.num_workers), pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=int(self.batch_size), shuffle=False,
            num_workers=int(self.num_workers), pin_memory=True
        )

        model = UNet(in_ch=3, out_ch=1, base=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        scaler = torch.cuda.amp.GradScaler(enabled=bool(self.amp))

        best = -1.0
        for ep in range(1, int(self.epochs) + 1):
            model.train()
            tl, td, n = 0.0, 0.0, 0
            for x, y, _sid in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(self.amp)):
                    logits = model(x)
                    loss = loss_bce_dice(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tl += float(loss.item())
                td += float(dice_score(logits.detach(), y).item())
                n += 1
            tl /= max(1, n); td /= max(1, n)

            model.eval()
            vl, vd, vn = 0.0, 0.0, 0
            saved = 0
            with torch.no_grad():
                for x, y, sid in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    logits = model(x)
                    loss = loss_bce_dice(logits, y)
                    vl += float(loss.item())
                    vd += float(dice_score(logits, y).item())
                    vn += 1

                    if self.debug_vis and saved < int(self.debug_vis):
                        save_debug_overlays(x[0], logits[0], exp / "debug_vis" / f"ep{ep:03d}_{saved}_{sid[0]}.png")
                        saved += 1
            vl /= max(1, vn); vd /= max(1, vn)

            print(f"Epoch {ep:03d} | train loss {tl:.4f} dice {td:.4f} | val loss {vl:.4f} dice {vd:.4f}")

            if vd > best:
                best = vd
                torch.save({"model": model.state_dict()}, exp / "ckpt" / "best.pt")

        print(f"[DONE] best_dice={best:.4f} exp_dir={exp}")
