#!/bin/sh
python standard_eval.py \
    --vsrl_annot_file ./data/vcoco/vcoco_test.json \
    --coco_file ./data/vcoco/instances_vcoco_all_2014.json \
    --split_file ./data/vcoco/vcoco_test.ids \
    --preds_file ./data/CDN/vcoco_r50.pickle