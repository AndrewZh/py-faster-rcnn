# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_radiuses = ex_rois[:, 2]
    ex_ctr_x = ex_rois[:, 0]
    ex_ctr_y = ex_rois[:, 1]

    gt_radiuses = gt_rois[:, 2]
    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_radiuses
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_radiuses
    targets_dr = np.log(gt_radiuses / ex_radiuses)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dr)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    radiuses = boxes[:, 2]
    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]

    dx = deltas[:, 0::3]
    dy = deltas[:, 1::3]
    dr = deltas[:, 2::3]

    pred_ctr_x = dx * radiuses[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * radiuses[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_r = np.exp(dr) * radiuses[:, np.newaxis]

    pred_circles = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_circles[:, 0::3] = pred_ctr_x
    # y1
    pred_circles[:, 1::3] = pred_ctr_y
    # r
    pred_circles[:, 2::3] = pred_r

    return pred_circles

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::3] = np.maximum(np.minimum(boxes[:, 0::3], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::3] = np.maximum(np.minimum(boxes[:, 1::3], im_shape[0] - 1), 0)
    # x1 < im_shape[1]
    boxes[:, 0::3] = np.maximum(np.minimum(boxes[:, 0::3], im_shape[1] - 1), 0)
    # y1 < im_shape[0]
    boxes[:, 1::3] = np.maximum(np.minimum(boxes[:, 1::3], im_shape[0] - 1), 0)
    # adjust radius
    x_dist = np.minimum(boxes[:, 1::3], im_shape[1] - boxes[:,0::3])
    y_dist = np.minimum(boxes[:, 0::3], im_shape[0] - boxes[:,1::3])
    boxes[:, 2::3] = np.minimum(boxes[:, 2::3], np.minimum(x_dist, y_dist))
    return boxes
