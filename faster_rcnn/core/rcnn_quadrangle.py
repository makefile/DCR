# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------
"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

from mxnet.context import current_context
import numpy as np
import numpy.random as npr

from bbox.bbox_transform import bbox_overlaps, bbox_transform
from bbox.rbbox_transform import bbox_transform_quadrangle
from bbox.bbox_regression import expand_bbox_regression_targets
from bbox.rbbox_regression import expand_bbox_regression_targets_quadrangle

from nms.nms_poly import cpu_polygon_overlaps, gpu_polygon_overlaps
from utils.dplog import Logger as logger
# from dataset.ds_utils import get_horizon_minAreaRectangle

def sample_rois_quadrangle(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None,
                           output_horizon_target=False, bbox_targets_h=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    pseudo quadrangle, the rois are indeed 2-point bbox
    :param rois: all_rois [n, 4]; e2e: [batch_idx, x1, y1, x2, y1, x2, y2, x1, y2]
    :param fg_rois_per_image: foreground roi number 32
    :param rois_per_image: total roi number 128
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, x2, y2, x3, y3, x4, y4, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    temp_rois = np.zeros((rois.shape[0], 5), dtype=rois.dtype)
    temp_rois[:, 0] = rois[:, 0]
    temp_rois[:, 1] = rois[:, 1]
    temp_rois[:, 2] = rois[:, 2]
    temp_rois[:, 3] = rois[:, 3]
    temp_rois[:, 4] = rois[:, 6]
    rois = temp_rois
    if labels is None:
        gt_boxes_bbox = np.zeros((gt_boxes.shape[0], 4), dtype=gt_boxes.dtype)
        ex_x = np.vstack((gt_boxes[:, 0], gt_boxes[:, 2], gt_boxes[:, 4], gt_boxes[:, 6]))
        ex_y = np.vstack((gt_boxes[:, 1], gt_boxes[:, 3], gt_boxes[:, 5], gt_boxes[:, 7]))
        gt_boxes_bbox[:, 0] = np.amin(ex_x, axis=0)
        gt_boxes_bbox[:, 1] = np.amin(ex_y, axis=0)
        gt_boxes_bbox[:, 2] = np.amax(ex_x, axis=0)
        gt_boxes_bbox[:, 3] = np.amax(ex_y, axis=0)
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes_bbox.astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 8]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform_quadrangle(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_quadrangle(bbox_target_data, num_classes, cfg)

    output_blob = [rois, labels, bbox_targets, bbox_weights]
    if output_horizon_target:
        # load or compute bbox_target
        if bbox_targets_h is not None:
            bbox_target_data_h = bbox_targets_h[keep_indexes, :]
        else:
            targets_h = bbox_transform(rois[:, 1:], gt_boxes_bbox[gt_assignment[keep_indexes]])
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                targets_h = ((targets_h - np.array(cfg.TRAIN.BBOX_MEANS[:4]))
                           / np.array(cfg.TRAIN.BBOX_STDS[:4]))
            bbox_target_data_h = np.hstack((labels[:, np.newaxis], targets_h))

        bbox_targets_h, bbox_weights_h = \
            expand_bbox_regression_targets(bbox_target_data_h, num_classes, cfg)
        output_blob.extend([bbox_targets_h, bbox_weights_h])

    return output_blob


def sample_rois_rotate(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    sample the quadrangle by polygon IOU
    :param rois: all_rois [n, 9]; e2e: [batch_idx, x1, y1, x2, y2, x3, y3, x4, y4]
    :param fg_rois_per_image: foreground roi number 32
    :param rois_per_image: total roi number 128
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, x2, y2, x3, y3, x4, y4, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """

    if labels is None:
        # logger.debug("pse start polygon_overlaps")
        # overlaps = cpu_polygon_overlaps(rois[:, 1:].astype(np.float32), gt_boxes[:, :8].astype(np.float32))
        overlaps = gpu_polygon_overlaps(rois[:, 1:].astype(np.float32), gt_boxes[:, :8].astype(np.float32), current_context().device_id)
        # logger.debug("pse end polygon_overlaps")
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 8]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform_quadrangle(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_quadrangle(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

