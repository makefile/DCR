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

"""
This file has functions about generating bounding box regression targets
"""

import numpy as np

def expand_bbox_regression_targets_quadrangle(bbox_targets_data, num_classes, cfg):
    """
    expand from 9 to 8 * num_classes; only the right class has non-zero bbox regression targets
    :param bbox_targets_data: [k * 9]
    :param num_classes: number of classes
    :return: bbox target processed [k * 8 num_classes]
    bbox_weights ! only foreground boxes have bbox regression computation!
    """
    classes = bbox_targets_data[:, 0]
    if cfg.CLASS_AGNOSTIC:
        num_classes = 2
    bbox_targets = np.zeros((classes.size, 8 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    indexes = np.where(classes > 0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(8 * 1 if cls > 0 else 0) if cfg.CLASS_AGNOSTIC else int(8 * cls)
        end = start + 8
        bbox_targets[index, start:end] = bbox_targets_data[index, 1:]
        bbox_weights[index, start:end] = cfg.TRAIN.BBOX_WEIGHTS
    return bbox_targets, bbox_weights

