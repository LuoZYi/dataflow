#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def check_one(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    cat_id_to_name = {
        int(c["id"]): str(c.get("name", f"category_{c['id']}"))
        for c in categories
    }

    ann_cat_counter = Counter(int(a["category_id"]) for a in annotations if "category_id" in a)
    image_ids = {int(img["id"]) for img in images if "id" in img}
    ann_image_ids = [int(a["image_id"]) for a in annotations if "image_id" in a]
    missing_image_ref = sum(1 for iid in ann_image_ids if iid not in image_ids)

    print("=" * 100)
    print(f"COCO: {path}")
    print(f"images:      {len(images)}")
    print(f"annotations: {len(annotations)}")
    print(f"categories:  {len(categories)}")

    if annotations:
        print("\nannotation counts by category:")
        for cid, n in sorted(ann_cat_counter.items()):
            print(f"  {cid:>4}  {cat_id_to_name.get(cid, 'UNKNOWN'):<30} {n}")

    if missing_image_ref:
        print(f"\n[WARN] annotations referencing missing image_id: {missing_image_ref}")

    if images:
        print("\nfirst image:")
        print(images[0])

    if annotations:
        print("\nfirst annotation summary:")
        a = annotations[0]
        summary = {
            "id": a.get("id"),
            "image_id": a.get("image_id"),
            "category_id": a.get("category_id"),
            "bbox": a.get("bbox"),
            "area": a.get("area"),
            "has_segmentation": "segmentation" in a,
        }
        print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_json", nargs="+", help="One or more COCO json files")
    args = parser.parse_args()

    for p in args.coco_json:
        check_one(Path(p))


if __name__ == "__main__":
    main()