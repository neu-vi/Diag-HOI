#!/bin/sh
python action_analysis/action_map_hicodet.py \
    --sum_gts_file ./data/hicodet/sum_gts_filtered.pkl \
    --gt_triplets_file ./data/hicodet/gt_wo_nointer.pkl \
    --preds_file ./data/CDN/preds_wact_sc_wo_nointer.pkl