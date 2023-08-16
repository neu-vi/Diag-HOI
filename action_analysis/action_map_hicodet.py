import numpy as np
import argparse
from collections import defaultdict
import os, tqdm, pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")))
from utils import *
# hico_valid_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

valid_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                              37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                              58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                              72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                              82, 84, 85, 86, 87, 88, 89, 90]
valid_verb_ids = list(range(1, 118))


def precompute_data(pred_hois, gt_hois, match_pairs, pred_bboxes, gt_bboxes, bbox_overlaps, gt_triplets, sum_gts_action):
    # find the matching between gt_hois and pred_hois
    # store each pred_hoi's error type: 1: human, 2: object, 3: both, 4: assoc, 5: action, 6: dup, 0: tp
    # find the best_matching_map
    pos_pred_ids = match_pairs.keys()
    matching_map = -np.ones(len(gt_hois)) # store the original matching between gt and pred. -1 means not pred can match a gt(missing case).
   
    pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True) 
    error_type_pred = -np.ones(len(pred_hois))

    for pred_hoi in pred_hois:
        is_match = 0
        flag = -1
        
        #triplet = (valid_obj_ids.index(pred_bboxes[pred_hoi['subject_id']]['category_id']), valid_obj_ids.index(pred_bboxes[pred_hoi['object_id']]['category_id']), valid_verb_ids.index(pred_hoi['category_id']))
        triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
            pred_sub_ids = match_pairs[pred_hoi['subject_id']]
            pred_obj_ids = match_pairs[pred_hoi['object_id']]

            pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
            pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
            pred_category_id = pred_hoi['category_id']
            
            max_overlap = 0
            max_gt_hoi = 0

            for gt_hoi in gt_hois:
                if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                    and pred_category_id == gt_hoi['category_id']:
                    is_match = 1
                    min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                         pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                    if min_overlap_gt > max_overlap:
                        max_overlap = min_overlap_gt
                        max_gt_hoi = gt_hoi
        
        # action error or association error: boxes are matched, is_match=0
        if  is_match==0 and len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
            for gt_hoi in gt_hois:
                if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids:
                    flag = 5
                    break
                    
            if flag == -1:
                flag = 4
                
        # both boxes correct, action duplicate
        if is_match == 1 and matching_map[gt_hois.index(max_gt_hoi)] != -1:
            flag = 6
            
        # human box correct, object box wrong
        if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] not in pos_pred_ids:
            flag = 2

        # object box correct, human box wrong
        if len(match_pairs) != 0 and pred_hoi['subject_id'] not in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
            flag = 1
            
        # both boxes wrong
        if len(match_pairs) == 0 or (pred_hoi['subject_id'] not in pos_pred_ids and pred_hoi['object_id'] not in pos_pred_ids):
            flag = 3
        
        if is_match == 1 and matching_map[gt_hois.index(max_gt_hoi)] == -1:
            flag = 0
            matching_map[gt_hois.index(max_gt_hoi)] = pred_hois.index(pred_hoi)
        error_type_pred[pred_hois.index(pred_hoi)] = flag
    
    best_matching_map = -np.ones(len(gt_hois)) # If fixed FP, what would be the best matching between gt and pred. -1 means not FP can be fixed to match a gt.
    for gt_hoi in gt_hois:
        for pred_hoi in pred_hois:
            if pred_hois.index(pred_hoi) in best_matching_map: # one pred can only match one gt.
                continue
            if pred_hois.index(pred_hoi) in matching_map:
                continue
            if matching_map[gt_hois.index(gt_hoi)] == pred_hois.index(pred_hoi): # already in matching_map
                break
            if len(match_pairs) != 0:
                if pred_hoi['subject_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    if gt_hoi['subject_id'] in pred_sub_ids: # human correct
                        if matching_map[gt_hois.index(gt_hoi)] > pred_hois.index(pred_hoi) or matching_map[gt_hois.index(gt_hoi)]==-1: # if the original TP has a lower score, or original matching cannot cover this gt
                            best_matching_map[gt_hois.index(gt_hoi)] = pred_hois.index(pred_hoi)
                            break
                elif pred_hoi['object_id'] in pos_pred_ids:
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    if gt_hoi['object_id'] in pred_obj_ids: # object correct
                        if matching_map[gt_hois.index(gt_hoi)] > pred_hois.index(pred_hoi) or matching_map[gt_hois.index(gt_hoi)]==-1: # if the original TP has a lower score, or original matching cannot cover this gt
                            best_matching_map[gt_hois.index(gt_hoi)] = pred_hois.index(pred_hoi)
                            break
    for i in range(len(matching_map)):
        gt_is_match = 0   
        gt_hoi = gt_hois[i]
        action_label = gt_hoi['category_id']
        for pred_hoi in pred_hois:
            if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                pred_obj_ids = match_pairs[pred_hoi['object_id']]
                if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids:
                    gt_is_match = 1
                    break
        if gt_is_match == 0:            
            sum_gts_action[action_label] -= 1

    return matching_map, error_type_pred, best_matching_map


def action_fptp(fp, tp, score, pred_hois, error_type_pred):

    pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True) 
    for pred_hoi in pred_hois:

        idx = pred_hois.index(pred_hoi)
        action_class = pred_hoi['category_id']
        # action_class = valid_verb_ids.index(pred_hoi['category_id'])

        if error_type_pred[idx] == 0:
            fp[action_class].append(0)
            tp[action_class].append(1)
            score[action_class].append(pred_hoi['action_score'])
        elif error_type_pred[idx] == 5 or error_type_pred[idx] == 6:
            fp[action_class].append(1)
            tp[action_class].append(0)
            score[action_class].append(pred_hoi['action_score'])
        

def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

def compute_map(sum_gts_action, gt_actions, fp, tp, score):

    ap = {}
    max_recall = {}

    for action in gt_actions:
        sum_gts_i = sum_gts_action[action]

        assert action != 57
           
        if sum_gts_i == 0:
            continue

        tp_i = np.array((tp[action]))
        fp_i = np.array((fp[action]))

        if len(tp_i) == 0:
            ap[action] = 0
            max_recall[action] = 0

        score_i = np.array(score[action])
        sort_inds = np.argsort(-score_i)

        fp_i = fp_i[sort_inds]
        tp_i = tp_i[sort_inds]

        fp_i = np.cumsum(fp_i)
        tp_i = np.cumsum(tp_i)
        
        rec = tp_i / sum_gts_i
        prec = tp_i / (fp_i + tp_i)

        if np.any(rec) == False:
          continue

        ap[action] = voc_ap(rec, prec)
        max_recall[action] = np.amax(rec)

    m_ap = np.mean(list(ap.values()))
    m_max_recall = np.mean(list(max_recall.values()))

    print('--------------------')
    print('mAP: {}  mean max recall: {}'.format(m_ap, m_max_recall))
    print('--------------------')

   
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum_gts_file', default='/work/vig/fangruiz/work/hoi_toolbox/data/HICO-Det/sum_gts_filtered.pkl')
    parser.add_argument('--gt_triplets_file', default='/work/vig/fangruiz/work/hoi_toolbox/data/HICO-Det/gt_triplets_wo_ni.pkl')  
    parser.add_argument('--preds_file', default='/work/vig/fangruiz/work/hoi_toolbox/data/HICO-Det/CDN/preds_wact_sc_wo_nointer.pkl')
   

    args = parser.parse_args()
    
    with open(args.preds_file, "rb") as f:
        predictions = pickle.load(f) # 8528

    with open(args.gt_triplets_file, "rb") as f:
        gts = pickle.load(f) # 8528

    with open(args.sum_gts_file, "rb") as f:
        sum_gts = pickle.load(f)
    

    sum_gts_action = {}
    gt_triplets = list(sum_gts.keys())
    for triplet in gt_triplets:
        action = triplet[2]
        if action not in sum_gts_action.keys():
            sum_gts_action[action] = sum_gts[triplet]
        else:
            sum_gts_action[action] += sum_gts[triplet]
    
    gt_actions = list(sum_gts_action.keys())

    fp = defaultdict(list)
    tp = defaultdict(list)
    score = defaultdict(list)

    for img_preds, img_gts in tqdm.tqdm(zip(predictions, gts)):

        pred_bboxes = img_preds['predictions']
        gt_bboxes = img_gts['annotations']
        pred_hois = img_preds['hoi_prediction']
        gt_hois = img_gts['hoi_annotation']

        if len(pred_hois)==0 or len(pred_bboxes)==0:
            continue
    
        bbox_pairs, bbox_overlaps = compute_iou_mat(gt_bboxes, pred_bboxes)

        matching_map, error_type_pred, best_matching_map = precompute_data(pred_hois, gt_hois, bbox_pairs, pred_bboxes, gt_bboxes, bbox_overlaps, gt_triplets, sum_gts_action)
        action_fptp(fp, tp, score, pred_hois, error_type_pred)
        
    compute_map(sum_gts_action, gt_actions, fp, tp, score)
    
    
    