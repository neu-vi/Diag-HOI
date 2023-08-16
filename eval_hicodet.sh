#!/bin/sh
python standard_eval.py \
    --sum_gts_file ./data/hicodet/sum_gts_filtered.pkl \
    --gt_triplets_file ./data/hicodet/gt_wo_nointer.pkl \
    --preds_file ./data/CDN/preds_wo_nointer.pkl 