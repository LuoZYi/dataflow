from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

def rle_decode(counts: List[int], h: int, w: int) -> np.ndarray:
    counts = [int(x) for x in counts]
    flat = np.zeros(h*w, dtype=np.uint8)
    idx=0; val=0
    for run in counts:
        if val==1 and run>0:
            flat[idx:idx+run]=1
        idx += run
        val ^= 1
    return flat.reshape((h,w), order="F").astype(bool)

def random_crop(img, mask, patch):
    H,W = img.shape[:2]
    if H < patch or W < patch:
        pad_h = max(0, patch-H)
        pad_w = max(0, patch-W)
        img = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
        mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode="constant")
        H,W = img.shape[:2]
    y = np.random.randint(0, H-patch+1)
    x = np.random.randint(0, W-patch+1)
    return img[y:y+patch, x:x+patch], mask[y:y+patch, x:x+patch]

def center_crop(img, mask, patch):
    H,W = img.shape[:2]
    if H < patch or W < patch:
        pad_h = max(0, patch-H)
        pad_w = max(0, patch-W)
        img = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
        mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode="constant")
        H,W = img.shape[:2]
    y = (H-patch)//2
    x = (W-patch)//2
    return img[y:y+patch, x:x+patch], mask[y:y+patch, x:x+patch]

class ParquetSemanticDataset(Dataset):
    def __init__(self, *, db_root: Path, ds_name: str, ds_root: Path, ann_name: str,
                 split: str, patch: int, train_mode: bool, seed: int = 0):
        self.ds_root = ds_root
        self.patch = int(patch)
        self.train_mode = bool(train_mode)

        meta = pd.read_parquet(db_root/ds_name/"meta.parquet")
        ann  = pd.read_parquet(db_root/ds_name/ann_name)

        sub = meta[meta["split"].astype(str) == split].copy()
        if len(sub) == 0:
            # fallback：如果 split 不存在/为空，就全用
            sub = meta.copy()
        self.meta = sub.reset_index(drop=True)

        ann["slide_uid"] = ann["slide_uid"].astype(str)
        self.ann_by_slide: Dict[str, List[dict]] = {}
        for r in ann.to_dict("records"):
            self.ann_by_slide.setdefault(str(r["slide_uid"]), []).append(r)

        self.slide_uids = self.meta["slide_uid"].astype(str).tolist()

        rng = np.random.RandomState(seed)
        rng.shuffle(self.slide_uids)

    def __len__(self):
        return len(self.slide_uids)

    def __getitem__(self, idx: int):
        slide_uid = self.slide_uids[idx]
        mr = self.meta[self.meta["slide_uid"].astype(str) == slide_uid].iloc[0].to_dict()
        ip = self.ds_root / str(mr["rel_path"])
        img = np.array(Image.open(ip).convert("RGB"))
        H,W = img.shape[:2]
        full = np.zeros((H,W), dtype=bool)

        for a in self.ann_by_slide.get(slide_uid, []):
            roi_x, roi_y, roi_w, roi_h = int(a["roi_x"]), int(a["roi_y"]), int(a["roi_w"]), int(a["roi_h"])
            mh, mw = int(a["rle_size_h"]), int(a["rle_size_w"])
            mroi = rle_decode(a["rle_counts"], mh, mw)
            y1 = min(H, roi_y+roi_h); x1 = min(W, roi_x+roi_w)
            yy = y1 - roi_y; xx = x1 - roi_x
            if yy>0 and xx>0:
                full[roi_y:y1, roi_x:x1] |= mroi[:yy, :xx]

        if self.train_mode:
            img, full = random_crop(img, full, self.patch)
        else:
            img, full = center_crop(img, full, self.patch)

        x = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        y = torch.from_numpy(full.astype(np.float32)).unsqueeze(0)
        return x, y, slide_uid
