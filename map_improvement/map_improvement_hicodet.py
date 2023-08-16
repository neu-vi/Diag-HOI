import numpy as np
import argparse, logging
from collections import defaultdict
import os, pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")))
from utils import *

def setup_logger(filepath):
  file_formatter = logging.Formatter(
      "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
      datefmt='%Y-%m-%d %H:%M:%S',
  )
  logger = logging.getLogger('example')
  handler = logging.StreamHandler()
  handler.setFormatter(file_formatter)
  logger.addHandler(handler)

  file_handle_name = "file"
  if file_handle_name in [h.name for h in logger.handlers]:
      return
  if os.path.dirname(filepath) != '':
      if not os.path.isdir(os.path.dirname(filepath)):
          os.makedirs(os.path.dirname(filepath))
  file_handle = logging.FileHandler(filename=filepath, mode="a")
  file_handle.set_name(file_handle_name)
  file_handle.setFormatter(file_formatter)
  logger.addHandler(file_handle)
  logger.setLevel(logging.DEBUG)
  return logger


def precompute_data(pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps, gt_triplets):
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
            if matching_map[gt_hois.index(gt_hoi)] == pred_hois.index(pred_hoi): # already in matching_map
                break
            if pred_hois.index(pred_hoi) in best_matching_map: # one pred can only match one gt.
                continue
            if pred_hois.index(pred_hoi) in matching_map:
                continue
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
          
    return matching_map, error_type_pred, best_matching_map


def fix_dup_or_both(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, error_type_pred, fix_type):

    pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True) 
    for pred_hoi in pred_hois:

        idx = pred_hois.index(pred_hoi)
        triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            if error_type_pred[idx] == 6 and fix_type == 'dup':
                continue
            if error_type_pred[idx] == 3 and fix_type == 'both':
                continue

            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])

def fix_none(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, error_type_pred, fix_type):

    pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True) 
    for pred_hoi in pred_hois:

        idx = pred_hois.index(pred_hoi)
        triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])


def fix_human(fp, tp, score, pred_hois, gt_hois, gt_bboxes, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map):

    added_tp_triplets = []
    added_score = []
    cnt = 0
    for i in range(len(matching_map)):
        
        pred_i = int(best_matching_map[i])
        if pred_i == -1: # cannot match any gt 
            continue
        ori_pred_i = int(matching_map[i])

        if error_type_pred[pred_i] == 1:
            pred_hois.append({
                'add': cnt,
                'score': pred_hois[pred_i]['score']}
            )
            triplet = (gt_bboxes[gt_hois[i]['subject_id']]['category_id'], gt_bboxes[gt_hois[i]['object_id']]['category_id'], gt_hois[i]['category_id'])
            added_tp_triplets.append(triplet)
            added_score.append(pred_hois[pred_i]['score'])
            error_type_pred = np.append(error_type_pred, [0])
            
            cnt += 1

            error_type_pred[pred_i] = -1
            if ori_pred_i != -1:
                error_type_pred[ori_pred_i] = -1 # set it to -1 to be suppressed during computing mAP
           
    scores_list = [k['score'] for k in pred_hois]
    sorted_idx = np.array(scores_list).argsort()[::-1]

    for idx in sorted_idx:

        pred_hoi = pred_hois[idx]
        if 'add' in pred_hoi:
            triplet = added_tp_triplets[pred_hoi['add']]
        else:
            triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            if error_type_pred[idx] == -1 or error_type_pred[idx] == 1:
                continue
            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])

def fix_object(fp, tp, score, pred_hois, gt_hois, gt_bboxes, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map):

    added_tp_triplets = []
    added_score = []
    cnt = 0
    for i in range(len(matching_map)):
        
        pred_i = int(best_matching_map[i])
        if pred_i == -1: # cannot match any gt 
            continue
        ori_pred_i = int(matching_map[i])

        if error_type_pred[pred_i] == 2:
            pred_hois.append({
                'add': cnt,
                'score': pred_hois[pred_i]['score']}
            )
            triplet = (gt_bboxes[gt_hois[i]['subject_id']]['category_id'], gt_bboxes[gt_hois[i]['object_id']]['category_id'], gt_hois[i]['category_id'])
            added_tp_triplets.append(triplet)
            added_score.append(pred_hois[pred_i]['score'])
            error_type_pred = np.append(error_type_pred, [0])
            
            cnt += 1

            error_type_pred[pred_i] = -1
            if ori_pred_i != -1:
                error_type_pred[ori_pred_i] = -1 # set it to -1 to be suppressed during computing mAP
           
    scores_list = [k['score'] for k in pred_hois]
    sorted_idx = np.array(scores_list).argsort()[::-1]

    for idx in sorted_idx:

        pred_hoi = pred_hois[idx]
        if 'add' in pred_hoi:
            triplet = added_tp_triplets[pred_hoi['add']]
        else:
            triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            if error_type_pred[idx] == -1 or error_type_pred[idx] == 2:
                continue
            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])

def fix_action(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map):

    added_tp_triplets = []
    added_score = []
    cnt = 0
    for i in range(len(matching_map)):
        
        pred_i = int(best_matching_map[i])
        if pred_i == -1: # cannot match any gt 
            continue
        ori_pred_i = int(matching_map[i])

        if error_type_pred[pred_i] == 5:
            pred_hois.append({
                'add': cnt,
                'score': pred_hois[pred_i]['score']}
            )
            triplet = (gt_bboxes[gt_hois[i]['subject_id']]['category_id'], gt_bboxes[gt_hois[i]['object_id']]['category_id'], gt_hois[i]['category_id'])
            added_tp_triplets.append(triplet)
            added_score.append(pred_hois[pred_i]['score'])
            error_type_pred = np.append(error_type_pred, [0])
            
            cnt += 1

            error_type_pred[pred_i] = -1
            if ori_pred_i != -1:
                error_type_pred[ori_pred_i] = -1 # set it to -1 to be suppressed during computing mAP
           
    scores_list = [k['score'] for k in pred_hois]
    sorted_idx = np.array(scores_list).argsort()[::-1]

    for idx in sorted_idx:

        pred_hoi = pred_hois[idx]
        if 'add' in pred_hoi:
            triplet = added_tp_triplets[pred_hoi['add']]
        else:
            triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            if error_type_pred[idx] == -1 or error_type_pred[idx] == 5:
                continue
            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])

def fix_assoc(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map):

    added_tp_triplets = []
    added_score = []
    cnt = 0
    for i in range(len(matching_map)):
        
        pred_i = int(best_matching_map[i])
        if pred_i == -1: # cannot match any gt 
            continue
        ori_pred_i = int(matching_map[i])

        if error_type_pred[pred_i] == 4:
            pred_hois.append({
                'add': cnt,
                'score': pred_hois[pred_i]['score']}
            )
            triplet = (gt_bboxes[gt_hois[i]['subject_id']]['category_id'], gt_bboxes[gt_hois[i]['object_id']]['category_id'], gt_hois[i]['category_id'])
            added_tp_triplets.append(triplet)
            added_score.append(pred_hois[pred_i]['score'])
            error_type_pred = np.append(error_type_pred, [0])
            
            cnt += 1

            error_type_pred[pred_i] = -1
            if ori_pred_i != -1:
                error_type_pred[ori_pred_i] = -1 # set it to -1 to be suppressed during computing mAP
           
    scores_list = [k['score'] for k in pred_hois]
    sorted_idx = np.array(scores_list).argsort()[::-1]

    for idx in sorted_idx:

        pred_hoi = pred_hois[idx]
        if 'add' in pred_hoi:
            triplet = added_tp_triplets[pred_hoi['add']]
        else:
            triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            if error_type_pred[idx] == -1 or error_type_pred[idx] == 4:
                continue
            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])

def fix_missing(fp, tp, score, gt_hois, pred_hois, gt_bboxes, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map, sum_gts):

    for i in range(len(matching_map)):
        pred_i = int(best_matching_map[i])
        ori_i = int(matching_map[i])
        triplet = (gt_bboxes[gt_hois[i]['subject_id']]['category_id'], gt_bboxes[gt_hois[i]['object_id']]['category_id'], gt_hois[i]['category_id'])
        if pred_i == -1 and ori_i == -1: # cannot match any gt 
            sum_gts[triplet] -= 1
           
    pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True) 
    for pred_hoi in pred_hois:
        
        idx = pred_hois.index(pred_hoi)
        
        triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
        
        assert triplet in gt_triplets

        if error_type_pred[idx] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
        else:
            fp[triplet].append(1)
            tp[triplet].append(0)
        score[triplet].append(pred_hoi['score'])

def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

def compute_map(sum_gts, rare_triplets, non_rare_triplets, gt_triplets, fp, tp, score, logger):

    ap = {}
    max_recall = {}
    rare_ap = {}
    non_rare_ap = {}

    rare_rec = {}
    non_rare_rec = {}

    for triplet in gt_triplets:
        sum_gts_i = sum_gts[triplet]

        assert triplet[2] != 57
           
        if sum_gts_i == 0:
            continue

        tp_i = np.array((tp[triplet]))
        fp_i = np.array((fp[triplet]))

        if len(tp_i) == 0:
            ap[triplet] = 0
            max_recall[triplet] = 0

        score_i = np.array(score[triplet])
        sort_inds = np.argsort(-score_i)

        fp_i = fp_i[sort_inds]
        tp_i = tp_i[sort_inds]

        fp_i = np.cumsum(fp_i)
        tp_i = np.cumsum(tp_i)
        
        rec = tp_i / sum_gts_i
        prec = tp_i / (fp_i + tp_i)

        if np.any(rec) == False:
          continue
        assert np.amax(rec) <= 1

        ap[triplet] = voc_ap(rec, prec)
        max_recall[triplet] = np.amax(rec)

        if triplet in rare_triplets:
            rare_ap[triplet] = ap[triplet]
            rare_rec[triplet] = max_recall[triplet]
        elif triplet in non_rare_triplets:
            non_rare_ap[triplet] = ap[triplet]
            non_rare_rec[triplet] = max_recall[triplet]
        else:
            print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        

    m_ap = np.mean(list(ap.values()))
    m_max_recall = np.mean(list(max_recall.values()))
    m_ap_rare = np.mean(list(rare_ap.values()))
    m_ap_non_rare = np.mean(list(non_rare_ap.values()))

    m_rec_rare = np.mean(list(rare_rec.values()))
    m_rec_non_rare = np.mean(list(non_rare_rec.values()))

    logger.info('--------------------')
    logger.info('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {} rare recall: {} non-rare recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall, m_rec_rare, m_rec_non_rare))
    logger.info('--------------------')

   
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum_gts_file', default=None)
    parser.add_argument('--gt_triplets_file', default=None)
    parser.add_argument('--preds_file', default=None)
    
    parser.add_argument('--fix_type', default=None, type=str)

    parser.add_argument('--rare_triplet_file', default=None)
    parser.add_argument('--non_rare_triplet_file', default=None)

    parser.add_argument('--model_name', default=None, type=str)
  
    parser.add_argument('--logger_path', default='./log', type=str)


    args = parser.parse_args()

    logger_name = os.path.join(args.logger_path, args.model_name, (args.fix_type+'.log'))

    logger = setup_logger(logger_name)

    
    with open(args.preds_file, "rb") as f:
        predictions = pickle.load(f) # 8528

    with open(args.gt_triplets_file, "rb") as f:
        gts = pickle.load(f) # 8528

    with open(args.sum_gts_file, "rb") as f:
        sum_gts = pickle.load(f)

    with open(args.rare_triplet_file, "rb") as f:
        rare_triplets = pickle.load(f)

    with open(args.non_rare_triplet_file, "rb") as f:
        non_rare_triplets = pickle.load(f)
     
    gt_triplets = list(sum_gts.keys())

    fp = defaultdict(list)
    tp = defaultdict(list)
    score = defaultdict(list)

    for img_preds, img_gts in zip(predictions, gts):

        pred_bboxes = img_preds['predictions']
        gt_bboxes = img_gts['annotations']
        pred_hois = img_preds['hoi_prediction']
        gt_hois = img_gts['hoi_annotation']
        
        if len(pred_hois)==0 or len(pred_bboxes)==0:
            continue
    
        bbox_pairs, bbox_overlaps = compute_iou_mat(gt_bboxes, pred_bboxes)
        
        matching_map, error_type_pred, best_matching_map = precompute_data(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps, gt_triplets)

        if args.fix_type == 'none':
            fix_none(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, error_type_pred, args.fix_type)
        if args.fix_type == 'dup' or args.fix_type == 'both':
            # duplication error or human-object boxes error
            fix_dup_or_both(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, error_type_pred, args.fix_type)
        if args.fix_type == 'hum':
            # human box error
            fix_human(fp, tp, score, pred_hois, gt_hois, gt_bboxes, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map)
        if args.fix_type == 'obj':
            # object box error
            fix_object(fp, tp, score, pred_hois, gt_hois, gt_bboxes, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map)
        if args.fix_type == 'action':
            # action error
            fix_action(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map)
        if args.fix_type == 'assoc':
            # association error
            fix_assoc(fp, tp, score, pred_hois, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map)
        if args.fix_type == 'missing':
            # missing GT 
            fix_missing(fp, tp, score, gt_hois, pred_hois, gt_bboxes, pred_bboxes, gt_triplets, matching_map, error_type_pred, best_matching_map, sum_gts)

    compute_map(sum_gts, rare_triplets, non_rare_triplets, gt_triplets, fp, tp, score, logger)
    
    
    