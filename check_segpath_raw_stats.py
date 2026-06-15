#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


IMAGE_SUFFIXES = ("_HE", "-HE", "_he", "-he")
MASK_SUFFIXES = ("_mask", "-mask", "_Mask", "-Mask", "_MASK", "-MASK")


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_manifest_files(subdir: Path):
    exts = {".csv", ".tsv", ".xlsx", ".xls"}
    return sorted(
        p for p in subdir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    )


def normalize_pair_key(filename: str) -> str:
    """
    ERG_Endothelium/ERG_Endothelium_024_148480_089088_HE.png
    ERG_Endothelium/ERG_Endothelium_024_148480_089088_mask.png
    -> ERG_Endothelium/ERG_Endothelium_024_148480_089088
    """
    p = Path(str(filename))
    parent = p.parent.as_posix()
    stem = p.stem

    for suf in IMAGE_SUFFIXES + MASK_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    return f"{parent}/{stem}" if parent not in {"", "."} else stem


def file_kind(filename: str) -> str:
    stem = Path(str(filename)).stem
    if any(stem.endswith(suf) for suf in IMAGE_SUFFIXES):
        return "image"
    if any(stem.endswith(suf) for suf in MASK_SUFFIXES):
        return "mask"
    return "unknown"


def find_existing_path(subdir: Path, filename: str) -> Path:
    p = subdir / filename
    if p.exists():
        return p

    # fallback: sometimes manifest path is relative to one nested root
    hits = list(subdir.rglob(Path(filename).name))
    if hits:
        return hits[0]

    return p


def process_subdataset(subdir: Path) -> dict:
    manifests = find_manifest_files(subdir)

    if not manifests:
        return {
            "subdataset": subdir.name,
            "manifest": None,
            "rows": 0,
            "images": 0,
            "masks": 0,
            "paired": 0,
            "image_only": 0,
            "mask_only": 0,
            "missing_files": 0,
            "split_counter": Counter(),
            "antibody_counter": Counter(),
            "note": "no manifest found",
        }

    # Prefer a file that has the needed columns
    chosen = None
    df = None
    for m in manifests:
        try:
            tmp = read_manifest(m)
        except Exception:
            continue

        cols = {c.lower().strip(): c for c in tmp.columns}
        if "filename" in cols:
            chosen = m
            df = tmp
            break

    if chosen is None or df is None:
        return {
            "subdataset": subdir.name,
            "manifest": str(manifests[0]),
            "rows": 0,
            "images": 0,
            "masks": 0,
            "paired": 0,
            "image_only": 0,
            "mask_only": 0,
            "missing_files": 0,
            "split_counter": Counter(),
            "antibody_counter": Counter(),
            "note": "manifest found but no filename column",
        }

    # normalize column names
    colmap = {c.lower().strip(): c for c in df.columns}
    filename_col = colmap["filename"]
    split_col = colmap.get("train_val_test", None)
    antibody_col = colmap.get("antibody", None)

    groups = defaultdict(lambda: {
        "images": [],
        "masks": [],
        "splits": [],
        "antibodies": [],
        "missing": 0,
    })

    total_images = 0
    total_masks = 0
    total_unknown = 0
    missing_files = 0

    for _, row in df.iterrows():
        filename = str(row[filename_col])
        if not filename or filename.lower() == "nan":
            continue

        kind = file_kind(filename)
        key = normalize_pair_key(filename)

        split = str(row[split_col]) if split_col is not None and pd.notna(row[split_col]) else "unspecified"
        antibody = str(row[antibody_col]) if antibody_col is not None and pd.notna(row[antibody_col]) else subdir.name

        full_path = find_existing_path(subdir, filename)
        exists = full_path.exists()

        if not exists:
            missing_files += 1
            groups[key]["missing"] += 1

        groups[key]["splits"].append(split)
        groups[key]["antibodies"].append(antibody)

        if kind == "image":
            total_images += 1
            groups[key]["images"].append(filename)
        elif kind == "mask":
            total_masks += 1
            groups[key]["masks"].append(filename)
        else:
            total_unknown += 1

    paired = 0
    image_only = 0
    mask_only = 0
    split_counter = Counter()
    antibody_counter = Counter()

    for key, g in groups.items():
        has_img = len(g["images"]) > 0
        has_mask = len(g["masks"]) > 0

        # one semantic sample is counted when both HE and mask exist
        if has_img and has_mask:
            paired += 1
            split_counter.update([g["splits"][0] if g["splits"] else "unspecified"])
            antibody_counter.update([g["antibodies"][0] if g["antibodies"] else subdir.name])
        elif has_img:
            image_only += 1
        elif has_mask:
            mask_only += 1

    return {
        "subdataset": subdir.name,
        "manifest": str(chosen),
        "rows": len(df),
        "images": total_images,
        "masks": total_masks,
        "unknown": total_unknown,
        "paired": paired,
        "image_only": image_only,
        "mask_only": mask_only,
        "missing_files": missing_files,
        "split_counter": split_counter,
        "antibody_counter": antibody_counter,
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/path_sam3/datasets/SegPath/SegPath")
    args = ap.parse_args()

    root = Path(args.root)

    subdirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    print("=" * 120)
    print("SegPath root:", root)
    print("subdatasets:", len(subdirs))

    total_paired = 0
    total_images = 0
    total_masks = 0
    total_missing = 0
    total_split_counter = Counter()
    total_antibody_counter = Counter()

    rows = []

    for subdir in subdirs:
        stat = process_subdataset(subdir)

        total_paired += stat["paired"]
        total_images += stat["images"]
        total_masks += stat["masks"]
        total_missing += stat["missing_files"]
        total_split_counter.update(stat["split_counter"])
        total_antibody_counter.update(stat["antibody_counter"])

        rows.append(stat)

        print("\n" + "-" * 120)
        print("subdataset:", stat["subdataset"])
        print("manifest:", stat["manifest"])
        print("rows:", stat["rows"])
        print("HE/images rows:", stat["images"])
        print("mask rows:", stat["masks"])
        print("paired image-mask samples:", stat["paired"])
        print("image-only groups:", stat["image_only"])
        print("mask-only groups:", stat["mask_only"])
        print("missing files:", stat["missing_files"])
        if stat["note"]:
            print("note:", stat["note"])

        if stat["split_counter"]:
            print("split counts:", dict(stat["split_counter"]))

        if stat["antibody_counter"]:
            print("antibody/category counts:", dict(stat["antibody_counter"]))

    print("\n" + "=" * 120)
    print("DATASET-LEVEL SUMMARY")
    print("raw HE/image rows:", total_images)
    print("raw mask rows:", total_masks)
    print("paired semantic samples/images:", total_paired)
    print("paired semantic annotations:", total_paired)
    print("missing files:", total_missing)
    print("split counts:", dict(total_split_counter))
    print("category/antibody counts:", dict(total_antibody_counter))
    print("num categories/antibodies:", len(total_antibody_counter))

    print("\nCompact table:")
    print("Dataset\tImages\tSemantic annotations\tCategories")
    print(f"SegPath\t{total_paired}\t{total_paired}\t{len(total_antibody_counter)}")


if __name__ == "__main__":
    main()