#!/bin/sh
python map_improvement/map_improvement_hicodet.py \
    --sum_gts_file ./data/hicodet/sum_gts_filtered.pkl \
    --gt_triplets_file ./data/hicodet/gt_wo_nointer.pkl \
    --rare_triplet_file ./data/hicodet/rare_triplets.pkl \
    --non_rare_triplet_file ./data/hicodet/non_rare_triplets.pkl \
    --model_name CDN \
    --logger_path ./log_hicodet \
    --preds_file ./data/CDN/preds_wo_nointer.pkl \
    --fix_type dup