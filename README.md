This repository contains scripts standardizing pathological datasets
- GlaS
- Lizard
- PaNNuke
- ConSep
- CRAG
- BCSS
including a range of 20x to 40x magnification in pathology, as well as tissues/glands and nuclei segmentation, in the format of both semantic and instance segmentation. 

Adapters contains dataset conversion scripts. 


**Run_parquet.py** is the main file for converting raw image and annotation files into a meta and annotation parquet.
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

