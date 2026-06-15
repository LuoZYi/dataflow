#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


IMAGE_SUFFIXES = ("_HE", "-HE", "_he", "-he")
MASK_SUFFIXES = ("_mask", "-mask", "_Mask", "-Mask", "_MASK", "-MASK")


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
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


def strip_known_suffix(stem: str) -> str:
    for suf in IMAGE_SUFFIXES + MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def normalize_pair_key(filename: str) -> str:
    p = Path(str(filename))
    parent = p.parent.as_posix()
    stem = strip_known_suffix(p.stem)
    return f"{parent}/{stem}" if parent not in {"", "."} else stem


def file_kind(filename: str) -> str:
    stem = Path(str(filename)).stem
    if any(stem.endswith(suf) for suf in IMAGE_SUFFIXES):
        return "image"
    if any(stem.endswith(suf) for suf in MASK_SUFFIXES):
        return "mask"
    return "unknown"


def find_existing_path(subdir: Path, filename: str) -> Path | None:
    p = subdir / filename
    if p.exists():
        return p

    hits = list(subdir.rglob(Path(filename).name))
    if hits:
        return hits[0]

    return None


def infer_he_from_mask_filename(mask_filename: str) -> str | None:
    p = Path(str(mask_filename))
    stem = p.stem

    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            return (p.parent / f"{base}_HE{p.suffix}").as_posix()

    return None


def process_subdataset(subdir: Path) -> dict:
    manifests = find_manifest_files(subdir)

    if not manifests:
        return {
            "subdataset": subdir.name,
            "manifest": None,
            "rows": 0,
            "image_samples": 0,
            "annotated_images": 0,
            "zero_ann_images": 0,
            "semantic_annotations": 0,
            "multi_ann_images": 0,
            "missing_files": 0,
            "split_image_counter": Counter(),
            "split_ann_counter": Counter(),
            "category_ann_counter": Counter(),
            "ann_per_image_dist": Counter(),
            "note": "no manifest found",
        }

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
            "image_samples": 0,
            "annotated_images": 0,
            "zero_ann_images": 0,
            "semantic_annotations": 0,
            "multi_ann_images": 0,
            "missing_files": 0,
            "split_image_counter": Counter(),
            "split_ann_counter": Counter(),
            "category_ann_counter": Counter(),
            "ann_per_image_dist": Counter(),
            "note": "manifest found but no filename column",
        }

    colmap = {c.lower().strip(): c for c in df.columns}
    filename_col = colmap["filename"]
    split_col = colmap.get("train_val_test")
    antibody_col = colmap.get("antibody")

    groups = defaultdict(lambda: {
        "images": set(),
        "masks": set(),
        "splits": [],
        "antibodies": [],
        "missing": 0,
    })

    missing_files = 0

    for _, row in df.iterrows():
        filename = str(row[filename_col])
        if not filename or filename.lower() == "nan":
            continue

        kind = file_kind(filename)
        key = normalize_pair_key(filename)

        split = (
            str(row[split_col]).strip()
            if split_col is not None and pd.notna(row[split_col])
            else "unspecified"
        )
        antibody = (
            str(row[antibody_col]).strip()
            if antibody_col is not None and pd.notna(row[antibody_col])
            else subdir.name
        )

        full_path = find_existing_path(subdir, filename)
        if full_path is None:
            missing_files += 1
            groups[key]["missing"] += 1

        groups[key]["splits"].append(split)
        groups[key]["antibodies"].append(antibody)

        if kind == "image":
            groups[key]["images"].add(filename)
        elif kind == "mask":
            groups[key]["masks"].add(filename)

            # Important fallback:
            # if manifest lists mask row but not HE row, try to infer paired HE path.
            he_guess = infer_he_from_mask_filename(filename)
            if he_guess is not None:
                he_path = find_existing_path(subdir, he_guess)
                if he_path is not None:
                    groups[key]["images"].add(he_guess)

    image_samples = 0
    annotated_images = 0
    zero_ann_images = 0
    semantic_annotations = 0
    multi_ann_images = 0

    split_image_counter = Counter()
    split_ann_counter = Counter()
    category_ann_counter = Counter()
    ann_per_image_dist = Counter()

    examples_multi = []
    examples_zero = []

    for key, g in groups.items():
        n_img = len(g["images"])
        n_mask = len(g["masks"])

        if n_img <= 0:
            # mask-only group without recoverable HE image; do not count as image sample.
            continue

        # Usually one key = one HE image. If duplicate HE rows exist, still count one sample.
        image_samples += 1

        split = g["splits"][0] if g["splits"] else "unspecified"
        antibody = g["antibodies"][0] if g["antibodies"] else subdir.name

        split_image_counter.update([split])
        ann_per_image_dist.update([n_mask])

        if n_mask == 0:
            zero_ann_images += 1
            if len(examples_zero) < 5:
                examples_zero.append(key)
        else:
            annotated_images += 1
            semantic_annotations += n_mask
            split_ann_counter.update([split] * n_mask)
            category_ann_counter.update([antibody] * n_mask)

        if n_mask > 1:
            multi_ann_images += 1
            if len(examples_multi) < 5:
                examples_multi.append((key, n_mask))

    return {
        "subdataset": subdir.name,
        "manifest": str(chosen),
        "rows": len(df),
        "image_samples": image_samples,
        "annotated_images": annotated_images,
        "zero_ann_images": zero_ann_images,
        "semantic_annotations": semantic_annotations,
        "multi_ann_images": multi_ann_images,
        "missing_files": missing_files,
        "split_image_counter": split_image_counter,
        "split_ann_counter": split_ann_counter,
        "category_ann_counter": category_ann_counter,
        "ann_per_image_dist": ann_per_image_dist,
        "examples_multi": examples_multi,
        "examples_zero": examples_zero,
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

    total_image_samples = 0
    total_annotated_images = 0
    total_zero_ann_images = 0
    total_semantic_annotations = 0
    total_multi_ann_images = 0
    total_missing_files = 0

    total_split_image_counter = Counter()
    total_split_ann_counter = Counter()
    total_category_ann_counter = Counter()
    total_ann_per_image_dist = Counter()

    print("=" * 120)
    print("SegPath root:", root)
    print("subdatasets:", len(subdirs))

    for subdir in subdirs:
        s = process_subdataset(subdir)

        total_image_samples += s["image_samples"]
        total_annotated_images += s["annotated_images"]
        total_zero_ann_images += s["zero_ann_images"]
        total_semantic_annotations += s["semantic_annotations"]
        total_multi_ann_images += s["multi_ann_images"]
        total_missing_files += s["missing_files"]

        total_split_image_counter.update(s["split_image_counter"])
        total_split_ann_counter.update(s["split_ann_counter"])
        total_category_ann_counter.update(s["category_ann_counter"])
        total_ann_per_image_dist.update(s["ann_per_image_dist"])

        print("\n" + "-" * 120)
        print("subdataset:", s["subdataset"])
        print("manifest:", s["manifest"])
        print("rows:", s["rows"])
        print("image samples:", s["image_samples"])
        print("annotated images:", s["annotated_images"])
        print("zero-annotation images:", s["zero_ann_images"])
        print("semantic annotations:", s["semantic_annotations"])
        print("multi-annotation images:", s["multi_ann_images"])
        print("missing files:", s["missing_files"])

        if s["note"]:
            print("note:", s["note"])

        print("annotation-per-image distribution:", dict(sorted(s["ann_per_image_dist"].items())))
        print("image split counts:", dict(s["split_image_counter"]))
        print("annotation split counts:", dict(s["split_ann_counter"]))
        print("category annotation counts:", dict(s["category_ann_counter"]))

        if s.get("examples_multi"):
            print("multi-ann examples:", s["examples_multi"])

        if s.get("examples_zero"):
            print("zero-ann examples:", s["examples_zero"])

    print("\n" + "=" * 120)
    print("DATASET-LEVEL SUMMARY")
    print("image samples:", total_image_samples)
    print("annotated images:", total_annotated_images)
    print("zero-annotation images:", total_zero_ann_images)
    print("semantic annotations:", total_semantic_annotations)
    print("multi-annotation images:", total_multi_ann_images)
    print("missing files:", total_missing_files)

    print("\nimage split counts:", dict(total_split_image_counter))
    print("annotation split counts:", dict(total_split_ann_counter))
    print("annotation-per-image distribution:", dict(sorted(total_ann_per_image_dist.items())))
    print("category annotation counts:", dict(total_category_ann_counter))
    print("num categories:", len(total_category_ann_counter))

    print("\nCompact table:")
    print("Dataset\tImages\tAnnotated images\tSemantic annotations\tCategories")
    print(
        f"SegPath\t{total_image_samples}\t{total_annotated_images}\t"
        f"{total_semantic_annotations}\t{len(total_category_ann_counter)}"
    )


if __name__ == "__main__":
    main()
