# import json, argparse
# from collections import Counter, defaultdict

# def load(p):
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)

# def key_fingerprint(lst, n=3):
#     ks = []
#     for i, x in enumerate(lst[:n]):
#         ks.append(tuple(sorted(x.keys())))
#     return ks

# def check(coco, name="coco"):
#     probs = []
#     imgs = coco.get("images", [])
#     anns = coco.get("annotations", [])
#     cats = coco.get("categories", [])

#     img_ids = [im.get("id") for im in imgs]
#     cat_ids = [c.get("id") for c in cats]
#     img_id_set = set(img_ids)
#     cat_id_set = set(cat_ids)

#     # missing required ann fields
#     req_ann = ["image_id", "category_id"]
#     miss = Counter()
#     miss_bbox = 0
#     bad_bbox = 0
#     bad_img_ref = 0
#     bad_cat_ref = 0

#     ann_img_ids = set()
#     for a in anns:
#         for r in req_ann:
#             if r not in a:
#                 miss[r] += 1
#         iid = a.get("image_id")
#         cid = a.get("category_id")
#         if iid not in img_id_set:
#             bad_img_ref += 1
#         else:
#             ann_img_ids.add(iid)
#         if cid not in cat_id_set:
#             bad_cat_ref += 1

#         if "bbox" not in a:
#             miss_bbox += 1
#         else:
#             bb = a["bbox"]
#             ok = isinstance(bb, (list, tuple)) and len(bb)==4
#             if not ok:
#                 bad_bbox += 1
#             else:
#                 x,y,w,h = bb
#                 if not all(isinstance(t,(int,float)) for t in bb) or w<=0 or h<=0:
#                     bad_bbox += 1

#     # images with zero ann
#     zero_ann = [iid for iid in img_id_set if iid not in ann_img_ids]

#     return {
#         "num_images": len(imgs),
#         "num_anns": len(anns),
#         "num_cats": len(cats),
#         "missing_ann_fields": dict(miss),
#         "missing_bbox": miss_bbox,
#         "bad_bbox": bad_bbox,
#         "bad_image_id_ref": bad_img_ref,
#         "bad_category_id_ref": bad_cat_ref,
#         "zero_annotation_images": len(zero_ann),
#         "zero_annotation_image_ids_first20": zero_ann[:20],
#         "img_key_fp": key_fingerprint(imgs),
#         "ann_key_fp": key_fingerprint(anns),
#         "cat_key_fp": key_fingerprint(cats),
#     }

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train", required=True)
#     ap.add_argument("--val", required=True)
#     args = ap.parse_args()

#     tr = load(args.train)
#     va = load(args.val)

#     print("=== Top-level keys ===")
#     print("train:", sorted(tr.keys()))
#     print("val  :", sorted(va.keys()))

#     print("\n=== Train summary ===")
#     trr = check(tr, "train")
#     for k,v in trr.items():
#         print(k, "=", v)

#     print("\n=== Val summary ===")
#     var = check(va, "val")
#     for k,v in var.items():
#         print(k, "=", v)

#     print("\n=== Fingerprint compare (should match) ===")
#     for field in ["img_key_fp","ann_key_fp","cat_key_fp"]:
#         print(field, "train=", trr[field], "val=", var[field])

# if __name__ == "__main__":
#     main()


# # parquet_sanity_quick.py
# import pandas as pd

# META = "/home/path_sam3/dataflow/parquet_db/v1/glas/meta.parquet"
# ANN  = "/home/path_sam3/dataflow/parquet_db/v1/glas/ann_instance.parquet"


# meta = pd.read_parquet(META)
# ann  = pd.read_parquet(ANN)

# # 1) ann.slide_uid 必须都能在 meta.slide_uid 找到
# missing_fk = ann.loc[~ann["slide_uid"].isin(meta["slide_uid"])]
# print("ann rows with missing meta slide_uid:", len(missing_fk))
# if len(missing_fk):
#     print(missing_fk["slide_uid"].head(20).tolist())

# # 2) 每张图 annotation 数
# cnt = ann.groupby("slide_uid").size()

# meta2 = meta[["slide_uid", "split", "rel_path", "width_px", "height_px", "backend_type"]].copy()
# meta2["ann_cnt"] = meta2["slide_uid"].map(cnt).fillna(0).astype(int)

# print("\nann_cnt by split:")
# print(meta2.groupby("split")["ann_cnt"].describe())

# zero = meta2[meta2["ann_cnt"] == 0].sort_values(["split","slide_uid"])
# print("\nZero-annotation images (by split):")
# print(zero[["split","slide_uid","rel_path","width_px","height_px","backend_type"]].head(200))
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

meta = pd.read_parquet("/home/path_sam3/dataflow/parquet_db/v1/GlaS/glas/meta.parquet")
ann  = pd.read_parquet("/home/path_sam3/dataflow/parquet_db/v1/GlaS/glas/ann.parquet")

# 0-annotation (based on parquet ann table)
cnt = ann.groupby("slide_uid").size()
meta["ann_cnt"] = meta["slide_uid"].map(cnt).fillna(0).astype(int)
zero = meta[meta["ann_cnt"]==0][["split","slide_uid","rel_path"]].copy()

roots = {
    "train": {
        "img": Path("/home/path_sam3/pipeline/datasets/Glas_new/train/images"),
        "lbl": Path("/home/path_sam3/pipeline/datasets/Glas_new/train/annos"),
    },
    "val": {
        "img": Path("/home/path_sam3/pipeline/datasets/Glas_new/val/images"),
        "lbl": Path("/home/path_sam3/pipeline/datasets/Glas_new/val/annos"),
    },
    "test": {
        "img": Path("/home/path_sam3/pipeline/datasets/Glas_new/test/images"),
        "lbl": Path("/home/path_sam3/pipeline/datasets/Glas_new/test/annos"),
    },
}

def label_path(split: str, rel_path: str) -> Path:
    # rel_path: e.g. testA_11.bmp -> testA_11_anno.bmp
    stem = Path(rel_path).stem
    return roots[split]["lbl"] / f"{stem}_anno.bmp"

def img_path(split: str, rel_path: str) -> Path:
    return roots[split]["img"] / rel_path

def exists_follow_symlink(p: Path) -> bool:
    # Path.exists() already follows symlink; this wrapper is just clarity
    return p.exists()

def inspect_mask(p: Path):
    im = Image.open(p)
    im.load()
    arr = np.array(im)
    return arr.shape, arr.dtype, int((arr != 0).sum()), int(arr.max())

print("Zero-annotation images (from ann parquet):", len(zero))

for _, r in zero.iterrows():
    split = r["split"]
    rp = r["rel_path"]
    ip = img_path(split, rp)
    lp = label_path(split, rp)

    # 1) image exists?
    if not ip.exists():
        print("[IMG MISSING]", split, rp, "->", ip)
        continue

    # 2) label exists (and symlink target exists)?
    if not exists_follow_symlink(lp):
        # 额外再看看“同名不带_anno”是不是误放了
        alt = roots[split]["lbl"] / rp
        print("[LBL MISSING]", split, rp, "expect", lp, "alt_same_name_exists=", alt.exists())
        continue

    # 3) label content nonzero?
    try:
        shape, dtype, nz, vmax = inspect_mask(lp)
        print("[LBL OK?]", split, rp, "mask_shape=", shape, "dtype=", dtype, "nonzero=", nz, "max=", vmax)
        if nz == 0:
            print("   -> mask is ALL ZERO (either truly empty, or wrong file)")
    except Exception as e:
        print("[LBL READ ERROR]", split, rp, "->", lp, "err=", repr(e))