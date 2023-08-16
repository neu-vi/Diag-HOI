# Partially modified from https://github.com/s-gupta/v-coco/blob/master/vsrl_eval.py
# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta
#
# This file is part of the VCOCO dataset hooks and is available
# under the terms of the Simplified BSD License provided in
# LICENSE. Please retain this notice and LICENSE if you use
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import pickle, os, json, copy, logging
import torch
import argparse
import numpy as np
from utils import *
from pycocotools.coco import COCO
from collections import defaultdict
import tqdm


def rec_prec_hicodet(args):
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
    gt_pairs_num = []
    TP_recall = 0
    TP_prec = 0

    # loop over images
    for img_gts in gts:
        image_name = img_gts['filename']

        if image_name not in pair_filenames:
            print(image_name)
            continue
        else:
            pair_idx = pair_filenames.index(image_name)
        pair_pred = pair_preds[pair_idx]
        box_pairs_pred = pair_pred['box_pairs']

        # object_labels = pair_pred['object_labels']
        gt_bboxes = img_gts['annotations']
        gt_hois = img_gts['hoi_annotation']

        # compute the number of pairs in gt for each image
        gt_pairs = []
        for k in range(len(gt_hois)):
            pair = (gt_hois[k]['subject_id'], gt_hois[k]['object_id'])
            if pair not in gt_pairs:
                gt_pairs.append(pair)
        gt_pairs_num.append(len(gt_pairs))

        # recall
        flag = torch.zeros(len(gt_pairs))
        flag_detail = -torch.ones(len(gt_pairs))
        for i in range(len(gt_pairs)):
            gt_h_box = gt_bboxes[gt_pairs[i][0]]
            gt_o_box = gt_bboxes[gt_pairs[i][1]]

            for j in range(len(box_pairs_pred)):
                pred_h_box = {'bbox': box_pairs_pred[j]['h_box'], 'category_id': 0}
                pred_o_box = {'bbox': box_pairs_pred[j]['o_box'], 'category_id': box_pairs_pred[j]['o_label']}
                # pred_o_box = {'bbox': box_pairs_pred[j]['o_box'], 'category_id': object_labels[j]}

                if compute_IOU(gt_h_box, pred_h_box) >=0.5 and compute_IOU(gt_o_box, pred_o_box) >=0.5:
                    if flag[i] == 0:
                        flag[i] = 1
                        flag_detail[i] = j
                    else:
                        break
        TP_recall += sum(flag)
        
        # precision
        num_pairs.append(len(box_pairs_pred))
        for i in range(len(box_pairs_pred)):
            if i in flag_detail:
                TP_prec += 1
        
    recall =  TP_recall/sum(gt_pairs_num)
    prec = TP_prec/sum(num_pairs)
    avg_num_pair = sum(num_pairs)/len(num_pairs)

    print(
        f"pair recall: {recall},"
        f"pair precision: {prec},"
        f"avg number of pairs: {avg_num_pair}."
    )

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

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

class VCOCOeval(object):

    def __init__(self, vsrl_annot_file, coco_annot_file,
        split_file):
        """Input:
        vslr_annot_file: path to the vcoco annotations
        coco_annot_file: path to the coco annotations
        split_file: image ids for split
        """
        self.COCO = COCO(coco_annot_file)
        self.VCOCO = _load_vcoco(vsrl_annot_file)
        self.image_ids = np.loadtxt(open(split_file, 'r'))
        # simple check
        assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), self.image_ids))

        self._init_coco()
        self._init_vcoco()


    def _init_vcoco(self):
        actions = [x['action_name'] for x in self.VCOCO]
        roles = [x['role_name'] for x in self.VCOCO]
        self.actions = actions
        self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
        self.num_actions = len(self.actions)
        self.roles = roles


    def _init_coco(self):
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}


    def _get_vcocodb(self):
        vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))
        for entry in vcocodb:
            self._prep_vcocodb_entry(entry)
            self._add_gt_annotations(entry)
        return vcocodb


    def _prep_vcocodb_entry(self, entry):
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['is_crowd'] = np.empty((0), dtype=bool)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
        entry['gt_role_id'] = np.empty((0, self.num_actions, 2), dtype=np.int32)


    def _add_gt_annotations(self, entry):
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_ann_ids = []
        width = entry['width']
        height = entry['height']
        for i, obj in enumerate(objs):
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
            x1, y1, x2, y2 = clip_xyxy_to_image(
                x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_ann_ids.append(ann_ids[i])
        num_valid_objs = len(valid_objs)
        assert num_valid_objs == len(valid_ann_ids)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
        gt_role_id = -np.ones((num_valid_objs, self.num_actions, 2), dtype=entry['gt_role_id'].dtype)

        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            is_crowd[ix] = obj['iscrowd']

            gt_actions[ix, :], gt_role_id[ix, :, :] = \
              self._get_vsrl_data(valid_ann_ids[ix],
                  valid_ann_ids, valid_objs)

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
        entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)


    def _get_vsrl_data(self, ann_id, ann_ids, objs):
        """ Get VSRL data for ann_id."""
        action_id = -np.ones((self.num_actions), dtype=np.int32)
        role_id = -np.ones((self.num_actions, 2), dtype=np.int32)
        # check if ann_id in vcoco annotations
        in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
        if in_vcoco.size > 0:
            action_id[:] = 0
            role_id[:] = -1
        else:
            return action_id, role_id
        for i, x in enumerate(self.VCOCO):
            assert x['action_name'] == self.actions[i]
            has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
            if has_label.size > 0:
                action_id[i] = 1
                assert has_label.size == 1
                rids = x['role_object_id'][has_label]
                assert rids[0, 0] == ann_id
                for j in range(1, rids.shape[1]):
                    if rids[0, j] == 0:
                        # no role
                        continue
                    aid = np.where(ann_ids == rids[0, j])[0]
                    assert aid.size > 0
                    role_id[i, j - 1] = aid
        return action_id, role_id

    # eval type: scenario_1
    def _do_pair_rec_prec(self, detections_file, logger):

        vcocodb = self._get_vcocodb()
        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)

        image_id_list = []
        for det in dets:
            image_id_list.append(det['image_id'])

        tp = 0
        num_gt_pair = []
        num_pred_pair = []
        for i in tqdm.tqdm(range(len(vcocodb))):
            image_id = vcocodb[i]['id']
            if image_id not in image_id_list:
                continue
            else:
                preds = dets[image_id_list.index(image_id)]
        
            all_boxes = vcocodb[i]['boxes']
            gt_human_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
            # person boxes
            gt_human_boxes = vcocodb[i]['boxes'][gt_human_inds]
            gt_actions = vcocodb[i]['gt_actions'][gt_human_inds]
            gt_roles_all = vcocodb[i]['gt_role_id']
            # some person instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore==True)[0]]==-1)

            gt_pairs = [] # (human idx, role idx, aid, rid)
            for i in range(len(gt_human_inds)):
                human_idx = gt_human_inds[i]
                aids = np.where(gt_actions[i]==1)[0]
                for aid in aids:
                    if len(self.roles[aid])<2:
                        continue
                    rids = np.arange(len(self.roles[aid])-1)
                    for rid in rids:
                        role_idx = gt_roles_all[human_idx, aid, rid]
                        if role_idx <= -1:
                            continue
                        else:
                            human_box = gt_human_boxes[i]
                            obj_box = all_boxes[gt_roles_all[human_idx, aid, rid]]
                            gt_pairs.append(
                              {
                                'h_box': human_box,
                                'o_box': obj_box
                              }
                            )
            flag = np.zeros(len(gt_pairs))

            for j in range(len(gt_pairs)):
                gt_h_box = gt_pairs[j]['h_box']
                gt_o_box = gt_pairs[j]['o_box']
                for pred_pair in preds['box_pairs']:
                    pred_h_box = pred_pair['h_box'][0].cpu().numpy()
                    pred_o_box = pred_pair['o_box'][0].cpu().numpy()
                   
                    if get_overlap(gt_h_box, pred_h_box)>=0.5 and get_overlap(gt_o_box, pred_o_box)>=0.5:
                        if flag[j] == 0:
                            flag[j] = 1
                        else:
                            break
            tp += np.sum(flag)
            num_gt_pair.append(len(gt_pairs))
            num_pred_pair.append(len(preds['box_pairs']))
            avg_num_pair = sum(num_pred_pair)/len(num_pred_pair)

        rec = tp/sum(num_gt_pair)
        prec = tp/sum(num_pred_pair)

        logger.info(
          f"pair recall: {rec},"
          f"pair precision: {prec},"
          f"avg number of pairs: {avg_num_pair}."
        )


def _load_vcoco(vcoco_file):
    print('loading vcoco annotations...')

    with open(vcoco_file, 'r') as f:
        vsrl_data = json.load(f)
    for i in range(len(vsrl_data)):
        vsrl_data[i]['role_object_id'] = \
        np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
        for j in ['ann_id', 'label', 'image_id']:
            vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
    return vsrl_data

def rec_prec_vcoco(args):

    logger_name = os.path.join(args.logger_path, (args.model_name+'.log'))
    logger = setup_logger(logger_name)

    vcocoeval = VCOCOeval(args.vsrl_annot_file, args.coco_file, args.split_file)
    vcocoeval._do_pair_rec_prec(args.model_det_file, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='hicodet')
    # hicodet data
    parser.add_argument('--gt_triplets_file', default=None)
    parser.add_argument('--pred_pair_file', default=None)

    # vcoco data
    parser.add_argument('--vsrl_annot_file', default='vcoco_test.json')
    parser.add_argument('--coco_file', default='instances_vcoco_all_2014.json')
    parser.add_argument('--split_file', default='vcoco_test.ids')
    parser.add_argument('--model_det_file', default=None)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--logger_path', default='./vcoco_log', type=str)



    args = parser.parse_args()

    if args.dataset == 'hicodet':
        rec_prec_hicodet(args)
    elif args.dataset == 'vcoco':
        rec_prec_vcoco(args)
    else:
        print('Do not support current dataset.')



    

    

    

    
    

    
    