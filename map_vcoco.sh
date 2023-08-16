#!/bin/sh
python map_improvement/map_improvement_vcoco.py \
    --vsrl_annot_file ./data/vcoco/vcoco_test.json \
    --coco_file ./data/vcoco/instances_vcoco_all_2014.json \
    --split_file ./data/vcoco/vcoco_test.ids \
    --model_det_file ./data/CDN/vcoco_r50.pickle \
    --fix_type dup \
    --save_inter_param_path ./inter_params_vcoco \
    --model_name CDN \
    --logger_path ./vcoco_log 
