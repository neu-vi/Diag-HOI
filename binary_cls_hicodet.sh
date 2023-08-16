#!/bin/sh
python action_analysis/binary_classification_hicodet.py \
    --pred_pair_file ./data/CDN/pair_pred_wscore.pkl \
    --gt_triplets_file ./data/hicodet/gt_wo_nointer.pkl

