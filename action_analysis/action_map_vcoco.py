# Modified from https://github.com/s-gupta/v-coco/blob/master/vsrl_eval.py
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

import numpy as np
from pycocotools.coco import COCO
import os, json
import copy
import pickle
import argparse, logging
from collections import defaultdict
import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")))
from utils import *

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


    def _collect_detections_for_image(self, dets, image_id):
        agents = np.empty((0, 4 + self.num_actions), dtype=np.float32)
        roles = np.empty((0, 5 * self.num_actions, 2), dtype=np.float32)
        # import ipdb; ipdb.set_trace()
        for det in dets:
            if det['image_id'] == image_id:
                # import ipdb; ipdb.set_trace()
                this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
                this_role  = np.zeros((1, 5 * self.num_actions, 2), dtype=np.float32)
                this_agent[0, :4] = det['person_box']
                for aid in range(self.num_actions):
                    for j, rid in enumerate(self.roles[aid]):
                        if rid == 'agent':
                            this_agent[0, 4 + aid] = det[self.actions[aid] + '_' + rid]
                        else:
                            this_role[0, 5 * aid: 5 * aid + 5, j-1] = det[self.actions[aid] + '_' + rid]
                agents = np.concatenate((agents, this_agent), axis=0)
                roles  = np.concatenate((roles, this_role), axis=0)
        return agents, roles


    def _do_eval(self, detections_file, fix_type, logger, model_param_path, model_name):
        vcocodb = self._get_vcocodb()

        self._do_role_eval(vcocodb, detections_file, fix_type, logger, model_param_path, model_name)


    def precompute_data(self, gt_all_boxes, gt_human_inds, gt_human_boxes, gt_actions, gt_roles_all, ignore, pred_agents, pred_roles, ovr_thresh=0.5):
        # pred_agents: N x 30 (4: box, 26: action scores)
        # pred_roles: N x 130 x 2(130: 5 x 26)
        # gt_roles_all: N_gt x 26 x 2
        # gt_all_boxes: N_gt x 4
        # gt_actions: N_hum x 26

        gt_hois = [] # (human idx, role idx, aid, rid)
        for i in range(len(gt_human_inds)):
            aids = np.where(gt_actions[i]==1)[0]
            for aid in aids:
                if len(self.roles[aid])<2:
                    continue
                rids = np.arange(len(self.roles[aid])-1)
                for rid in rids:
                    role_idx = gt_roles_all[gt_human_inds[i], aid, rid]
                    gt_hois.append((gt_human_inds[i], role_idx, aid, rid))
        matching_map = -np.ones(len(gt_hois)) # store the original matching between gt and pred. -1 means not pred can match a gt(missing case).

        error_type_pred = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc_pred = [[[] for r in range(2)] for a in range(self.num_actions)]

        pred_agent_boxes_list = []
        pred_role_boxes_list = []
        pred_scores_list = []
        pred_idx_list = []
        flag_list = []
        flag_idx_list = []
        all_idx = 0
        for aid in range(self.num_actions):
            if len(self.roles[aid])<2:
                # if action has no role, then no role AP computed
                continue

            for rid in range(len(self.roles[aid])-1):

                # keep track of detected instances for each action for each role
                covered = np.zeros((gt_human_boxes.shape[0]), dtype=bool)

                # get gt roles for action and role
                gt_role_inds = gt_roles_all[gt_human_inds, aid, rid]
                gt_roles = -np.ones_like(gt_human_boxes)
                for j in range(gt_human_boxes.shape[0]):
                    if gt_role_inds[j] > -1:
                        gt_roles[j] = gt_all_boxes[gt_role_inds[j]]

                # import ipdb; ipdb.set_trace()
                agent_boxes = pred_agents[:, :4]
                role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
                agent_scores = pred_roles[:, 5 * aid + 4, rid]

                valid = np.where(np.isnan(agent_scores) == False)[0]
                agent_scores = agent_scores[valid]
                agent_boxes = agent_boxes[valid, :]
                role_boxes = role_boxes[valid, :]

                pred_agent_boxes_list.append(agent_boxes)
                pred_role_boxes_list.append(role_boxes)
                # pred_scores_list.append(agent_scores)
                pred_idx_list.append(valid)

                idx = agent_scores.argsort()[::-1]

                for j in idx: # loop over all predictions
                    # store each pred_hoi's error type: 1: human, 2: object, 3: both, 4: assoc, 5: action, 6: dup, 0: tp, -1: not used
                    flag = -1
                    pred_box = agent_boxes[j, :]
                    overlaps = get_overlap(gt_human_boxes, pred_box)

                    # matching happens based on the person
                    jmax = overlaps.argmax()
                    ovmax = overlaps.max()


                    pred_scores_list.append(agent_scores[j])
                    sc_pred[aid][rid].append(agent_scores[j])
                    flag_idx_list.append((aid, rid, list(idx).index(j)))
                    # if matched with an instance with no annotations
                    # continue
                    if ignore[jmax]:
                        error_type_pred[aid][rid].append(flag) # append -1
                        flag_list.append(flag)
                        all_idx += 1
                        continue

                    else: # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1): # if no gt role
                            if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                # if no role is predicted, mark it as correct role overlap
                                ov_role = 1.0
                            else:
                                # if a role is predicted, mark it as false
                                ov_role = 0.0
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])
                      
                        is_true_action = (gt_actions[jmax, aid] == 1)

                        if is_true_action and (ovmax>=ovr_thresh) and (ov_role>=ovr_thresh):
                            if covered[jmax]:
                                flag = 6
                            else:
                                if np.all(gt_roles[jmax, :] == -1):
                                    matching_map[gt_hois.index((gt_human_inds[jmax], -1, aid, rid))] = all_idx
                                else:
                                    matching_map[gt_hois.index((gt_human_inds[jmax], gt_role_inds[jmax], aid, rid))] = all_idx
                                flag = 0
                                covered[jmax] = True
                        elif (ovmax>=ovr_thresh) and (ov_role>=ovr_thresh):
                            flag = 5
                        elif ovmax>=ovr_thresh:
                            if not np.all(np.isnan(role_boxes[j, :])):
                                other_overlaps = get_overlap(gt_all_boxes, role_boxes[j, :])
                                if other_overlaps.max() >= ovr_thresh:
                                    flag = 4
                                else:
                                    flag = 2
                            else:
                                flag = 2
                        elif ov_role>=ovr_thresh:
                            if not np.all(np.isnan(pred_box)):
                                other_overlaps = get_overlap(gt_all_boxes, pred_box)
                                if other_overlaps.max() >= ovr_thresh:
                                    flag = 4
                                else:
                                    flag = 1
                            else:
                                flag = 1
                        else:
                            flag = 3
                        all_idx += 1
                        flag_list.append(flag)
                        error_type_pred[aid][rid].append(flag)

        return matching_map, error_type_pred, pred_scores_list, gt_hois, flag_list, flag_idx_list, sc_pred

    # eval type: scenario_1
    def _do_role_eval(self, vcocodb, detections_file, logger, model_param_path, model_name):

        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)

        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]


        npos = np.zeros((self.num_actions, 2), dtype=np.float32)

        inter_param_path = os.path.join(model_param_path, (model_name+'.pkl'))
        if not os.path.exists(inter_param_path):
            inter_param = defaultdict(list)
            is_exist = 0
        else:
            is_exist = 1
            with open(inter_param_path, 'rb') as f:
                inter_param = pickle.load(f)


        for i in tqdm.tqdm(range(len(vcocodb))):
            image_id = vcocodb[i]['id']
        
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

            for aid in range(self.num_actions):
                for rid in range(len(self.roles[aid])-1):
                    npos[aid][rid] += np.sum(gt_actions[:, aid] == 1)

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)

            if is_exist == 0:
                if i==0:
                    logger.info('compute inter params')
                matching_map, error_type_pred, pred_scores_list, gt_hois, flag_list, flag_idx_list, sc_pred = \
                  self.precompute_data(all_boxes, gt_human_inds, gt_human_boxes, gt_actions, gt_roles_all, ignore, pred_agents, pred_roles)
                inter_param['matching_map'].append(matching_map)
                inter_param['error_type_pred'].append(error_type_pred)
                inter_param['pred_scores_list'].append(pred_scores_list)
                inter_param['gt_hois'].append(gt_hois)
                inter_param['flag_list'].append(flag_list)
                inter_param['flag_idx_list'].append(flag_idx_list)
                inter_param['sc_pred'].append(sc_pred)
                if i==len(vcocodb)-1:
                    with open(inter_param_path, "wb") as f:
                        pickle.dump(inter_param, f)
            else:
                if i==0:
                    logger.info('load inter params')
                matching_map = inter_param['matching_map'][i]
                error_type_pred = inter_param['error_type_pred'][i]
                pred_scores_list = inter_param['pred_scores_list'][i]
                gt_hois = inter_param['gt_hois'][i]
                flag_list = inter_param['flag_list'][i]
                flag_idx_list = inter_param['flag_idx_list'][i]
                sc_pred = inter_param['sc_pred'][i]

            
            self.fix_none(fp, tp, sc, error_type_pred, pred_scores_list)

        self.compute_map(fp, tp, sc, npos, logger)

        print("finish")

    def compute_map(self, fp, tp, sc, npos, logger):
        # compute ap for each action
        role_ap = np.zeros((self.num_actions, 2), dtype=np.float32)
        role_ap[:] = np.nan

        for aid in range(self.num_actions):
            if len(self.roles[aid])<2:
                continue
            for rid in range(len(self.roles[aid])-1):
                a_fp = np.array(fp[aid][rid], dtype=np.float32)
                a_tp = np.array(tp[aid][rid], dtype=np.float32)
                a_sc = np.array(sc[aid][rid], dtype=np.float32)

                # sort in descending score order
                idx = a_sc.argsort()[::-1]
                a_fp = a_fp[idx]
                a_tp = a_tp[idx]
                a_sc = a_sc[idx]

                a_fp = np.cumsum(a_fp)
                a_tp = np.cumsum(a_tp)

                rec = a_tp / float(npos[aid][rid])
                #check
                if np.any(rec) == False:
                    role_ap[aid, rid] = 0.0
                    continue
                if float(npos[aid][rid])==0.0:
                    role_ap[aid, rid] = 0.0
                    continue

                assert(np.amax(rec) <= 1)
                prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
                role_ap[aid, rid] = voc_ap(rec, prec)
        logger.info('---------Reporting Role AP (%)------------------')
        for aid in range(self.num_actions):
            if len(self.roles[aid])<2: continue
            for rid in range(len(self.roles[aid])-1):
                logger.info('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid]+'-'+self.roles[aid][rid+1], role_ap[aid, rid]*100.0, int(npos[aid][rid])))
        logger.info('Average Role [%s] AP = %.2f'%('scenario 1', np.nanmean(role_ap) * 100.00))
        logger.info('---------------------------------------------')

    def fix_none(self, fp, tp, sc, error_type_pred, pred_scores_list):
        cnt = 0
        for aid in range(self.num_actions):
            if len(self.roles[aid])<2:
                # if action has no role, then no role AP computed
                continue
            for rid in range(len(self.roles[aid])-1):
                for flag in error_type_pred[aid][rid]:
                    if flag == -1:
                        cnt += 1
                        continue
                    elif flag == 0:
                        tp[aid][rid].append(1)
                        fp[aid][rid].append(0)
                        sc[aid][rid].append(pred_scores_list[cnt])
                    elif flag == 5 or flag == 6:
                        fp[aid][rid].append(1)
                        tp[aid][rid].append(0)
                        sc[aid][rid].append(pred_scores_list[cnt])

                    cnt += 1
        return fp, tp, sc


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


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    [as defined in PASCAL VOC]
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--vsrl_annot_file', default='vcoco_test.json')
    parser.add_argument('--coco_file', default='instances_vcoco_all_2014.json')
    parser.add_argument('--split_file', default='vcoco_test.ids')

    parser.add_argument('--model_det_file', default=None)
    
    parser.add_argument('--save_inter_param_path', default=None)
    parser.add_argument('--model_name', default='CDN', type=str)

    parser.add_argument('--logger_path', default=None, type=str)


    args = parser.parse_args()

    logger_name = os.path.join(args.logger_path, (args.model_name+'.log'))

    logger = setup_logger(logger_name)

    vcocoeval = VCOCOeval(args.vsrl_annot_file, args.coco_file, args.split_file)

    vcocoeval._do_eval(args.model_det_file, logger, args.save_inter_param_path, args.model_name)
