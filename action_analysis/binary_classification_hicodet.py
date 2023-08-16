import pickle
import argparse
import numpy as np
import tqdm, os, sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")))
from utils import *

def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.01):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 110.
    return ap


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_triplets_file', default=None)
    parser.add_argument('--pred_pair_file', default=None)
    
    args = parser.parse_args()

    with open(args.gt_triplets_file, "rb") as f:
        gts = pickle.load(f) # 8528

    with open(args.pred_pair_file, "rb") as f:
        pair_preds = pickle.load(f) # 9546
  
    # the length of gts and pair_preds are different, 
    # because no_interaction has been removed from gts.
    
    pair_filenames = []
    for i in range(len(pair_preds)):
        pair_filenames.append(pair_preds[i]['filename'])

    
    num_pairs = []
    tp = []
    fp = []
    sc = []
    gt = 0
    pred=0
    
    # loop over images
    for img_gts in tqdm.tqdm(gts):
        image_name = img_gts['filename']

        pair_idx = pair_filenames.index(image_name)
        pair_pred = pair_preds[pair_idx]
        box_pairs_pred = pair_pred['box_pairs']
        # object_labels = pair_pred['object_labels']
        pred += len(box_pairs_pred)

        gt_bboxes = img_gts['annotations']
        gt_hois = img_gts['hoi_annotation']

        # compute the number of pairs in gt for each image
        gt_triplets = []
        for k in range(len(gt_hois)):
            triplet = (gt_hois[k]['subject_id'], gt_hois[k]['object_id'], gt_hois[k]['category_id'])
            gt_triplets.append(triplet)

       
        for i in range(len(box_pairs_pred)):
            flag = 0
            pred_h_box = {'bbox': box_pairs_pred[i]['h_box'], 'category_id': 0}
            # pred_o_box = {'bbox': box_pairs_pred[i]['o_box'], 'category_id': box_pairs_pred[i]['o_label']}
            pred_o_box = {'bbox': box_pairs_pred[i]['o_box'], 'category_id': box_pairs_pred[i]['o_label']}
            # no_inter_score = 0.1
            no_inter_score = 1- box_pairs_pred[i]['hoi_score_inter']

            for j in range(len(gt_triplets)):
                gt_h_box = gt_bboxes[gt_triplets[j][0]]
                gt_o_box = gt_bboxes[gt_triplets[j][1]]
                # breakpoint()
                if compute_IOU(gt_h_box, pred_h_box) >=0.5 and compute_IOU(gt_o_box, pred_o_box) >=0.5:
                    flag = 1
                    break
            
            if flag == 0:
                gt += 1
                tp.append(1)
                fp.append(0)
            else:
                fp.append(1)
                tp.append(0)
            sc.append(no_inter_score)
        
    tp = np.array(tp)
    fp = np.array(fp)
    score = np.array(sc)
    sort_inds = np.argsort(-score)

    fp = fp[sort_inds]
    tp = tp[sort_inds]

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / gt
    prec = tp / (tp+fp)

    assert np.amax(rec)<=1
    ap = voc_ap(rec, prec)
       
    print(
        f"no interaction ap: {ap},"
    )

    
    

    
    
