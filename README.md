This repository contains scripts standardizing pathological datasets

- GlaS
- Lizard
- PaNNuke
- ConSep
- CRAG
- BCSS
  
including a range of 20x to 40x magnification in pathology, as well as tissues/glands and nuclei segmentation, in the format of both semantic and instance segmentation. 

Adapters contains dataset conversion scripts. 


**run_parquet.py** is the main file for converting raw image and annotation files into a meta and annotation parquet.
usage: run_parquet.py [-h] --out_dir OUT_DIR [--input INPUT] [--dataset DATASET] [--root ROOT] [--out_layout {per_dataset,merged}] [--max_samples MAX_SAMPLES]
                      [--chunk_size CHUNK_SIZE] [--compression COMPRESSION] [--include_aug INCLUDE_AUG] [--bcss_split_mode {unspecified,random}] [--seed SEED]
                      [--val_ratio VAL_RATIO] [--test_ratio TEST_RATIO] [--mpp MPP] [--mpp_x MPP_X] [--mpp_y MPP_Y] [--rle_roi {full,bbox}] [--store_extra]
                      [--emit_semantic_from_instance EMIT_SEMANTIC_FROM_INSTANCE] [--semantic_mode {per_label,all}] [--semantic_min_area SEMANTIC_MIN_AREA]
                      [--ann_out_mode {mixed,split}]


**validate_dataset.py** can be used to verify the sanity of the datasets
usage: validate_dataset.py [-h] --dataset {consep,crag,bcss,glas,lizard,pannuke} --root ROOT [--max_samples MAX_SAMPLES] [--fail_fast] [--verbose] [--dump_json DUMP_JSON]
                           [--include_aug INCLUDE_AUG] [--bcss_split_mode {unspecified,random}] [--seed SEED] [--val_ratio VAL_RATIO] [--test_ratio TEST_RATIO]


**parquet_to_coco.py** converts parquet annotation to coco.json format so that it can be used for SAM3
usage: parquet_to_coco.py [-h] --meta META --ann ANN --img_root IMG_ROOT --out OUT [--split SPLIT] [--ann_kind {instance,semantic,all}] [--full_rle FULL_RLE]
                          [--drop_missing_files DROP_MISSING_FILES] [--max_images MAX_IMAG]


To run training for GlaS, use ********glas_instance_train_updated.coco.json** as the training annotation file and **glas_instance_val_updated.coco.json** as the validation annotation file.

The rest of the files used can be found in the config files.








## `eval_whole_image.py`

### What it does
`eval_whole_image.py` evaluates a **patch-trained SAM3 model on whole (original) images**.  
It runs **tiled/sliding-window inference** on each full-resolution test image, **maps patch predictions back to the original image coordinates**, **merges duplicates caused by overlapping tiles** (so the same nucleus is not counted multiple times), and finally computes **COCO instance segmentation metrics (segm AP/AR)** against the **original image-level COCO ground truth**.

This is the correct way to get *“before patch / image-level”* metrics when training was done on patches.

### Inputs / Outputs
- **Input**: original (whole-image) COCO GT JSON (e.g., `consep_test_instance.coco.json`) + original image folder
- **Input**: base SAM3 checkpoint + your finetuned checkpoint
- **Output**: a COCO-format prediction JSON on whole images (merged), and printed COCOeval summary

### Key arguments
- `--coco_json`: image-level COCO GT json (whole images)
- `--img_root`: root directory so that `img_root / file_name` exists
- `--base_ckpt`: official/base `sam3.pt` (pretrained)
- `--ft_ckpt`: your finetuned checkpoint from training
- `--out_json`: output path for merged whole-image predictions (COCO results json)
- `--tile`: tile/patch size used for sliding-window inference (e.g., 384)
- `--overlap`: overlap between tiles (e.g., 96)
- `--conf`: confidence threshold (lower = more detections, higher = fewer)
- `--nms_iou`: optional global NMS IoU threshold as a safety net (e.g., 0.5; set <=0 to disable)
- `--max_dets`: **max predictions per whole image** used for truncation and COCOeval `maxDets`
  - For dense nuclei datasets like CoNSeP, use **1000–3000** to avoid artificially low recall.

### Example command
```bash
python /home/path_sam3/dataflow/eval_whole_image.py \
  --coco_json /home/path_sam3/dataflow/generated_coco/consep/consep_test_instance.coco.json \
  --img_root  /home/path_sam3/pipeline/data_links/CoNSeP/CoNSeP \
  --base_ckpt /home/path_sam3/.hf/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt \
  --ft_ckpt   /home/path_sam3/dataflow/runs/consep/consep_patch/checkpoints/checkpoint.pt \
  --out_json  /home/path_sam3/dataflow/runs/consep/consep_patch/preds_whole.json \
  --tile 384 --overlap 96 --conf 0.01 --nms_iou 0.5 --max_dets 3000
```


## `make_patch_coco.py`


### What it does
`make_patch_coco.py` converts an **image-level COCO instance segmentation dataset** into a **patch-based COCO dataset**.

For each original image, it:
1. Generates a sliding-window grid of patches (with configurable `patch_size` and `overlap`).
2. Crops and saves each patch image to `out_dir/images/`.
3. For each patch, finds all COCO instances whose **bbox intersects** the patch region.
4. Crops each instance segmentation (RLE/polygon) to the patch, then recomputes:
   - patch-local `segmentation`
   - patch-local `bbox` (xywh)
   - patch-local `area`
5. Writes a new patch COCO file to `out_dir/annotations/instances_patch.json`.
6. Saves a `patch_map.jsonl` file that records the mapping from each patch back to its source image
   (useful for later whole-image reconstruction / debugging).

This is the standard offline preprocessing step when the original images contain **too many instances**
(e.g., CoNSeP nuclei often > 1000/image) and SAM3 has a fixed `num_queries` budget.

### Inputs / Outputs
- **Input**
  - `--coco_json`: the original **whole-image** COCO annotation json
  - `--img_root`: directory such that `img_root / file_name` exists for each COCO image
- **Output**
  - `out_dir/images/` : patch PNGs
  - `out_dir/annotations/instances_patch.json` : patch-level COCO GT
  - `out_dir/patch_map.jsonl` : patch→source mapping info

### Key arguments
- `--coco_json` *(required)*: path to the original image-level COCO json
- `--img_root` *(required)*: root directory for images referenced by `file_name` in COCO
- `--out_dir` *(required)*: output directory for patch dataset
- `--patch_size`: patch size in pixels (default often 384)
- `--overlap`: overlap in pixels between adjacent patches (default often 96)
- `--min_area`: drop tiny cropped fragments whose patch-local area is smaller than this (e.g., 20)
- `--max_gt_per_patch`: hard cap on GT instances per patch (safety for dense patches; e.g., 120–180)
- `--keep_empty_patches`: if set, keep patches with 0 GT; otherwise empty patches are dropped

### Example: create patch dataset for CoNSeP train split
```bash
python make_patch_coco.py \
  --coco_json /home/path_sam3/dataflow/generated_coco/consep/consep_train_instance.coco.json \
  --img_root  /home/path_sam3/pipeline/data_links/CoNSeP/CoNSeP \
  --out_dir   /home/path_sam3/dataflow/generated_coco/consep_patches_train_dup \
  --patch_size 384 --overlap 96 --min_area 20 --max_gt_per_patch 120
```

/dataflow_train not in use

/configs contains training configurations of SAM 3. It is recommended to use local pretrained SAM 3 official checkpoint instead of loading from hf. Simply put the config file under /sam3/sam3/train/configs and run the official SAM 3 train.py using that particular config file.

Cheers!