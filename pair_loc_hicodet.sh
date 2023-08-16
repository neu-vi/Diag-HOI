#!/bin/sh
python pair_localization.py \
    --dataset hicodet \
    --gt_triplets_file ./data/hicodet/gt_wo_nointer.pkl \
    --pred_pair_file ./data/CDN/hicodet_pair_preds.pkl