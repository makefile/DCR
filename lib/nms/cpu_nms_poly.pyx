# --------------------------------------------------------
# C++ implementation of polygon IOU, by fyk
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import numpy as np
cimport numpy as np


cdef extern from "cpu_polyiou.cxx":
    double _cpu_iou_poly(np.float32_t* p, np.float32_t* q)

def poly_overlaps_cython(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 8) ndarray of float
    query_boxes: (K, 8) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] box_i, box_j
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype=np.float32)
    cdef unsigned int k, n
    for k in range(K):
        for n in range(N):
            box_i, box_j = boxes[n, :8], query_boxes[k, :8]
            overlaps[n, k] = _cpu_iou_poly(&box_i[0], &box_j[0])
            # overlaps[n, k] = np.random.random()
    return overlaps


def cpu_nms_poly(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] box_i, box_j
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 8]

    # np.intp in python is same as ssize_t in C/C++, well the *_t is only used for `cdef`
    # see https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython
    cdef np.ndarray[np.intp_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j

    cdef np.float32_t ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)

        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue

            box_i, box_j = dets[i, :8], dets[j, :8]
            ovr = _cpu_iou_poly(&box_i[0], &box_j[0])
            # ovr = np.random.random()
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
