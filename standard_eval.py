import numpy as np
import argparse, pickle, copy, json
from pycocotools.coco import COCO
from collections import defaultdict
from utils import *

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
    
        for det in dets:
            if det['image_id'] == image_id:
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


    def _do_eval(self, detections_file, ovr_thresh=0.5):
        vcocodb = self._get_vcocodb()
        self._do_role_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh, eval_type='scenario_1')

  
    def _do_role_eval(self, vcocodb, detections_file, ovr_thresh=0.5, eval_type='scenario_1'):

        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)
    
        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)

        for i in range(len(vcocodb)):
      
            image_id = vcocodb[i]['id']
      
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]
            gt_actions = vcocodb[i]['gt_actions'][gt_inds]
            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore==True)[0]]==-1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(gt_actions[:, aid] == 1)

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)
      
            for aid in range(self.num_actions):
                if len(self.roles[aid])<2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid])-1): 
                    # keep track of detected instances for each action for each role
                    covered = np.zeros((gt_boxes.shape[0]), dtype=bool)

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid, rid]
                    gt_roles = -np.ones_like(gt_boxes)
                    for j in range(gt_boxes.shape[0]):
                        if gt_role_inds[j] > -1:
                        gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
                    agent_scores = pred_roles[:, 5 * aid + 4, rid]

                    valid = np.where(np.isnan(agent_scores) == False)[0]
                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    idx = agent_scores.argsort()[::-1]

                    for j in idx:
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_boxes, pred_box)
                        # num_qual = len(np.where(overlaps>=0.5)[0])
                        # matching happens based on the person 
                        jmax = overlaps.argmax()
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1): # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false 
                                    ov_role = 0.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0   
                            else:
                                raise ValueError('Unknown eval type')    
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])
                        # get_overlap(gt_roles[2, :].reshape((1, 4)), role_boxes[1, :])
                        is_true_action = (gt_actions[jmax, aid] == 1)

                        # import ipdb; ipdb.set_trace()
                        sc[aid][rid].append(agent_scores[j])
                        if is_true_action and (ovmax>=ovr_thresh) and (ov_role>=ovr_thresh):
                            if covered[jmax]:  
                                fp[aid][rid].append(1)
                                tp[aid][rid].append(0)
                            else:
                                fp[aid][rid].append(0)  
                                tp[aid][rid].append(1)
                                covered[jmax] = True
                        else:
                            fp[aid][rid].append(1)
                            tp[aid][rid].append(0)
             
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

        # import ipdb; ipdb.set_trace()
        # sort in descending score order
        idx = a_sc.argsort()[::-1]
        a_fp = a_fp[idx]
        a_tp = a_tp[idx]
        a_sc = a_sc[idx]

        a_fp = np.cumsum(a_fp)
        a_tp = np.cumsum(a_tp)

        rec = a_tp / float(npos[aid])
        #check
        assert(np.amax(rec) <= 1)
        prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
        role_ap[aid, rid] = voc_ap(rec, prec)
    # import ipdb; ipdb.set_trace()
    print('---------Reporting Role AP (%)------------------')
    for aid in range(self.num_actions):
      if len(self.roles[aid])<2: continue
      for rid in range(len(self.roles[aid])-1):
        print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid]+'-'+self.roles[aid][rid+1], role_ap[aid, rid]*100.0, int(npos[aid])))
    print('Average Role [%s] AP = %.2f'%(eval_type, np.nanmean(role_ap) * 100.00))  
    print('---------------------------------------------') 

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

def compute_fptp(fp, tp, score, pred_hois, gt_hois, match_pairs, pred_bboxes, gt_bboxes, bbox_overlaps, gt_triplets):

    pos_pred_ids = match_pairs.keys()
    vis_tag = np.zeros(len(gt_hois))
    pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True) 
    
    for pred_hoi in pred_hois:

        is_match = 0

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
            
        if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
            fp[triplet].append(0)
            tp[triplet].append(1)
            vis_tag[gt_hois.index(max_gt_hoi)] = 1
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

def compute_map(sum_gts, gt_triplets, fp, tp, score):

    ap = {}
    max_recall = {}

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
        
        ap[triplet] = voc_ap(rec, prec)
        max_recall[triplet] = np.amax(rec)
       
    m_ap = np.mean(list(ap.values()))
    m_max_recall = np.mean(list(max_recall.values()))

    print('--------------------')
    print('mAP: {}  mean max recall: {}'.format(m_ap, m_max_recall))
    print('--------------------')

def eval_hicodet(args):

    with open(args.preds_file, "rb") as f:
        predictions = pickle.load(f) # 8528

    with open(args.gt_triplets_file, "rb") as f:
        gts = pickle.load(f) # 8528

    with open(args.sum_gts_file, "rb") as f:
        sum_gts = pickle.load(f)
    
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
        compute_fptp(fp, tp, score, pred_hois, gt_hois, bbox_pairs, pred_bboxes, gt_bboxes, bbox_overlaps, gt_triplets)
    
    compute_map(sum_gts, gt_triplets, fp, tp, score)

def eval_vcoco(args):
    vcocoeval = VCOCOeval(args.vsrl_annot_file, args.coco_file, args.split_file)
    vcocoeval._do_eval(args.preds_file, ovr_thresh=0.5)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hicodet', help='eval dataset, hicodet/vcoco')
                        
    # HICO-Det precomputed data
    parser.add_argument('--sum_gts_file', default=None, help='ground truth triplets and the number of ground truth for each triplet')
    parser.add_argument('--gt_triplets_file', default=None, help='ground truth hois for each image in HICO-Det')

    # V-COCO precomputed data
    parser.add_argument('--vsrl_annot_file', default=None)
    parser.add_argument('--coco_file', default=None)
    parser.add_argument('--split_file', default=None)

    # prediction file
    parser.add_argument('--preds_file', default=None, help='prediction file')


    args = parser.parse_args()

    if args.dataset == 'hicodet':
        eval_hicodet(args)
    elif args.dataset == 'vcoco':
        eval_vcoco(args)
    else:
        print('Do not support current dataset.')
    
    
    
    