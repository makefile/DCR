# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Modified by fyk
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

"""
Generate base anchors on index 0
"""

import numpy as np
from math import radians, cos, sin
from dataset.ds_utils import get_best_begin_point

def generate_rotate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6),
                     angles=[-60, -30, 0, 30, 60, 90]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales X angles wrt a reference (0,0, 15,0, 15,15, 0,15) window.
    """

    base_anchor = np.array([1, 1, base_size, 1, base_size, base_size, 1, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    scale_anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                               for i in range(ratio_anchors.shape[0])])
    anchors = np.vstack([_angle_enum(scale_anchors[i, :], angles) for i in range(scale_anchors.shape[0])])

    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[4] - anchor[0] + 1
    h = anchor[5] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    xa = x_ctr - 0.5 * (ws - 1)
    ya = y_ctr - 0.5 * (hs - 1)
    xc = x_ctr + 0.5 * (ws - 1)
    yc = y_ctr + 0.5 * (hs - 1)
    # clockwise 4 point
    anchors = np.hstack((xa, ya,
                         xc, ya,
                         xc, yc,
                         xa, yc))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _add_angle(anchor, angle):
    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # code from OpenCV: void RotatedRect::points(Point2f pt[]) const
    # RotatedRect degree range -90~0, so if degree > 0, swap width with height
    if angle > 0: w, h = h, w
    _angle = radians(angle)

    b = cos(_angle) * 0.5
    a = sin(_angle) * 0.5

    pt0x = x_ctr - a * h - b * w
    pt0y = y_ctr + b * h - a * w
    pt1x = x_ctr + a * h - b * w
    pt1y = y_ctr - b * h - a * w
    pt2x = 2 * x_ctr - pt0x
    pt2y = 2 * y_ctr - pt0y
    pt3x = 2 * x_ctr - pt1x
    pt3y = 2 * y_ctr - pt1y
    anchor = [pt0x, pt0y, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y]

    return np.array(get_best_begin_point(anchor))


def _angle_enum(anchor, angles):
    """
    Enumerate a set of anchors for each angle wrt an anchor.
    """
    anchors = np.vstack([_add_angle(anchor, angles[i]) for i in range(len(angles))])
    return anchors

def validate_clockwise_points(points):
    """
    Validates that the 4 points that a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory > 0:
        return False
    else:
        return True

if __name__ == '__main__':
    a = generate_rotate_anchors()
    for x in a:
        if not validate_clockwise_points(x): print 'no clockwise'
    # a = generate_rotate_anchors(base_size=8, ratios=[0.25,0.4])
    # from IPython import embed; embed()
    for i in range(len(a)):
        print("%6d, %6d, %6d, %6d, %6d, %6d, %6d, %6d"%
              (a[i,0],a[i,1],a[i,2],a[i,3],a[i,4],a[i,5],a[i,6],a[i,7]))
