#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPTION_MAP = {
    "ERG_Endothelium": "ERG-positive endothelial regions.",
    "panCK_Epithelium": "panCK-positive epithelial regions.",
    "CD45RB_Leukocyte": "CD45RB-positive leukocyte regions.",
    "CD3CD20_Lymphocyte": "CD3/CD20-positive lymphocyte regions.",
    "MNDA_MyeloidCell": "MNDA-positive myeloid cell regions.",
    "MIST1_PlasmaCell": "MIST1-positive plasma cell regions.",
    "CD235a_RBC": "CD235a-positive red blood cell regions.",
    "aSMA_SmoothMuscle": "aSMA-positive smooth muscle regions.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_txt", default=None)
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args()

    in_json = Path(args.in_json)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    coco = json.loads(in_json.read_text(encoding="utf-8"))

    cat_id_to_name = {
        int(c["id"]): str(c.get("name", f"category_{c['id']}"))
        for c in coco.get("categories", [])
    }

    images_by_id = {
        int(im["id"]): im
        for im in coco.get("images", [])
    }

    semantic_captions = []
    txt_lines = [
        "\t".join([
            "source_json",
            "caption_id",
            "image_id",
            "image_file_name",
            "category_id",
            "category_name",
            "mask_file_name",
            "simple_caption",
            "complex_caption",
            "errors",
        ])
    ]

    for ann in coco.get("annotations", []):
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])

        category_name = cat_id_to_name.get(category_id, f"category_{category_id}")
        image_file_name = str(images_by_id[image_id]["file_name"])

        caption = CAPTION_MAP.get(
            category_name,
            f"{category_name.replace('_', ' ')} regions."
        )

        caption_id = f"{in_json.stem}:{ann_id}"

        ann["semantic_caption_id"] = caption_id
        ann["caption"] = caption
        ann["caption_simple"] = caption
        ann["caption_complex"] = caption

        images_by_id[image_id].setdefault("semantic_caption_ids", []).append(caption_id)

        record = {
            "id": caption_id,
            "image_id": image_id,
            "image_file_name": image_file_name,
            "annotation_id": ann_id,
            "category_id": category_id,
            "category_name": category_name,
            "mask_file_name": "",
            "mask_path": "",
            "captions": {
                "simple": caption,
                "complex": caption,
            },
        }

        semantic_captions.append(record)

        txt_lines.append("\t".join([
            in_json.name,
            caption_id,
            str(image_id),
            image_file_name,
            str(category_id),
            category_name,
            "",
            caption,
            caption,
            "",
        ]))

    coco["semantic_captions"] = semantic_captions
    coco["caption_generation"] = {
        "generator": "deterministic_category_template",
        "dataset_name": "SegPath",
        "source_coco_json": str(in_json),
        "annotation_caption_fields": ["caption", "caption_simple", "caption_complex"],
        "note": "SegPath masks are often tiny/sparse marker targets; category-level text is used instead of VLM-generated morphology captions.",
    }

    out_json.write_text(
        json.dumps(coco, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )

    if args.out_txt:
        out_txt = Path(args.out_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        print("[OK] wrote txt:", out_txt)

    print("[OK] wrote captioned coco:", out_json)
    print("images:", len(coco.get("images", [])))
    print("annotations:", len(coco.get("annotations", [])))
    print("semantic_captions:", len(semantic_captions))


if __name__ == "__main__":
    main()
