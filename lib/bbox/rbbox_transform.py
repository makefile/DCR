# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Modified by fyk
# --------------------------------------------------------
# Based on:
# py-faster-rcnn
# Copyright (c) 2016 by Contributors
# Licence under The MIT License
# py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import numpy as np

def clip_quadrangle_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_scale: calculate afterwards bbox coordinates, [height, width]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # boxes[:, 0::8] = np.round(boxes[:, 0::8].copy() * im_scale[1])
    # boxes[:, 2::8] = np.round(boxes[:, 2::8].copy() * im_scale[1])
    # boxes[:, 4::8] = np.round(boxes[:, 4::8].copy() * im_scale[1])
    # boxes[:, 6::8] = np.round(boxes[:, 6::8].copy() * im_scale[1])
    # boxes[:, 1::8] = np.round(boxes[:, 1::8].copy() * im_scale[0])
    # boxes[:, 3::8] = np.round(boxes[:, 3::8].copy() * im_scale[0])
    # boxes[:, 5::8] = np.round(boxes[:, 5::8].copy() * im_scale[0])
    # boxes[:, 7::8] = np.round(boxes[:, 7::8].copy() * im_scale[0])
    # x1 >= 0
    boxes[:, 0::8] = np.maximum(np.minimum(boxes[:, 0::8], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::8] = np.maximum(np.minimum(boxes[:, 1::8], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::8] = np.maximum(np.minimum(boxes[:, 2::8], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::8] = np.maximum(np.minimum(boxes[:, 3::8], im_shape[0] - 1), 0)
    # x1 >= 0
    boxes[:, 4::8] = np.maximum(np.minimum(boxes[:, 4::8], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 5::8] = np.maximum(np.minimum(boxes[:, 5::8], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 6::8] = np.maximum(np.minimum(boxes[:, 6::8], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 7::8] = np.maximum(np.minimum(boxes[:, 7::8], im_shape[0] - 1), 0)
    return boxes

def nonlinear_transform_quadrangle(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 8]
    :return: [N, 8]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    # ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    # ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    # ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    # ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)
    #
    # gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    # gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    # gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    # gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)
    #
    # targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    # targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    # targets_dw = np.log(gt_widths / ex_widths)
    # targets_dh = np.log(gt_heights / ex_heights)
    #
    # targets = np.vstack(
    #     (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    ex_x1 = ex_rois[:, 0]
    ex_y1 = ex_rois[:, 1]
    ex_x2 = ex_rois[:, 2]
    ex_y2 = ex_rois[:, 3]

    ex_width = ex_x2 - ex_x1 + 1.0
    ex_height = ex_y2 - ex_y1 + 1.0

    gt_rois.astype(float)

    # print 'in transform', ex_width, ex_height

    targets_dx1 = (gt_rois[:, 0] - ex_x1) / (ex_width + 1e-14)
    targets_dx2 = (gt_rois[:, 2] - ex_x2) / (ex_width + 1e-14)
    targets_dx3 = (gt_rois[:, 4] - ex_x2) / (ex_width + 1e-14)
    targets_dx4 = (gt_rois[:, 6] - ex_x1) / (ex_width + 1e-14)
    targets_dy1 = (gt_rois[:, 1] - ex_y1) / (ex_height + 1e-14)
    targets_dy2 = (gt_rois[:, 3] - ex_y1) / (ex_height + 1e-14)
    targets_dy3 = (gt_rois[:, 5] - ex_y2) / (ex_height + 1e-14)
    targets_dy4 = (gt_rois[:, 7] - ex_y2) / (ex_height + 1e-14)

    # print 'in transform calculate delta'
    # print gt_rois[:, 0] - ex_x1, gt_rois[:, 1] - ex_y1, gt_rois[:, 2] - ex_x2, gt_rois[:, 3] - ex_y1, gt_rois[:, 4] - ex_x2, gt_rois[:, 5] - ex_y2, gt_rois[:, 6] - ex_x1, gt_rois[:, 7]

    targets = np.vstack((targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4)).transpose()
    return targets

def nonlinear_pred_quadrangle(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 8 * num_classes]
    :return: [N 8 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    widths = widths[:, np.newaxis]
    heights = heights[:, np.newaxis]
    x1 = boxes[:, 0][:, np.newaxis]
    y1 = boxes[:, 1][:, np.newaxis]
    x2 = boxes[:, 2][:, np.newaxis]
    y2 = boxes[:, 1][:, np.newaxis]
    x3 = boxes[:, 2][:, np.newaxis]
    y3 = boxes[:, 3][:, np.newaxis]
    x4 = boxes[:, 0][:, np.newaxis]
    y4 = boxes[:, 3][:, np.newaxis]

    # print 'in pred quadrangle', x1, y1, x2, y2, x3, y3, x4, y4

    dx1 = box_deltas[:, 0::8]
    dy1 = box_deltas[:, 1::8]
    dx2 = box_deltas[:, 2::8]
    dy2 = box_deltas[:, 3::8]
    dx3 = box_deltas[:, 4::8]
    dy3 = box_deltas[:, 5::8]
    dx4 = box_deltas[:, 6::8]
    dy4 = box_deltas[:, 7::8]

    # print 'in pred quadrangle target', dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4

    # pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    # pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # pred_w = np.exp(dw) * widths[:, np.newaxis]
    # pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::8] = x1 + dx1 * widths
    # y1
    pred_boxes[:, 1::8] = y1 + dy1 * heights
    # x2
    pred_boxes[:, 2::8] = x2 + dx2 * widths
    # y2
    pred_boxes[:, 3::8] = y2 + dy2 * heights
    # x3
    pred_boxes[:, 4::8] = x3 + dx3 * widths
    # y3
    pred_boxes[:, 5::8] = y3 + dy3 * heights
    # x4
    pred_boxes[:, 6::8] = x4 + dx4 * widths
    # y4
    pred_boxes[:, 7::8] = y4 + dy4 * heights

    # print 'after pred', dx1 * widths, dy1 * heights, dx2 * widths, dy2 * heights, dx3 * widths, dy3 * heights, dx4 * widths, dy4 * heights

    return pred_boxes

# define bbox_transform and bbox_pred
bbox_pred_quadrangle = nonlinear_pred_quadrangle
bbox_transform_quadrangle = nonlinear_transform_quadrangle
