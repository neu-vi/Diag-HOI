#!/bin/sh
python action_analysis/action_map_vcoco.py \
    --vsrl_annot_file ./data/vcoco/vcoco_test.json \
    --coco_file ./data/vcoco/instances_vcoco_all_2014.json \
    --split_file ./data/vcoco/vcoco_test.ids \
    --model_det_file ./data/CDN/vcoco_act_results.pkl \
    --save_inter_param_path ./inter_params_vcoco \
    --model_name CDN \
    --logger_path ./vcoco_log 