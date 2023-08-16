import numpy as np

def compute_IOU(bbox1, bbox2):

    if isinstance(bbox1['category_id'], str):
        bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
    if isinstance(bbox2['category_id'], str):
        bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
    if bbox1['category_id'] == bbox2['category_id']:
        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
        S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

        sum_area = S_rec1 + S_rec2

        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
            return intersect / (sum_area - intersect)
    else:
        return 0

def compute_iou_mat(bbox_list1, bbox_list2, overlap_iou=0.5):

    iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
    if len(bbox_list1) == 0 or len(bbox_list2) == 0:
        return {}
    for i, bbox1 in enumerate(bbox_list1):
        for j, bbox2 in enumerate(bbox_list2):
            iou_i = compute_IOU(bbox1, bbox2)
            iou_mat[i, j] = iou_i
   
    iou_mat_ov=iou_mat.copy()
    iou_mat[iou_mat>=overlap_iou] = 1
    iou_mat[iou_mat<overlap_iou] = 0
    
    match_pairs = np.nonzero(iou_mat)
    match_pairs_dict = {}
    match_pair_overlaps = {}
    if iou_mat.max() > 0:
        for i, pred_id in enumerate(match_pairs[1]):
            if pred_id not in match_pairs_dict.keys():
                match_pairs_dict[pred_id] = []
                match_pair_overlaps[pred_id]=[]
            match_pairs_dict[pred_id].append(match_pairs[0][i])
            match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
    return match_pairs_dict, match_pair_overlaps

def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2

def get_overlap(boxes, ref_box):
    ixmin = np.maximum(boxes[:, 0], ref_box[0])
    iymin = np.maximum(boxes[:, 1], ref_box[1])
    ixmax = np.minimum(boxes[:, 2], ref_box[2])
    iymax = np.minimum(boxes[:, 3], ref_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
            (boxes[:, 2] - boxes[:, 0] + 1.) *
            (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps