import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms_poly.hpp":
    void _poly_nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)
    void _poly_overlaps(np.float32_t*, np.float32_t*, np.float32_t*, int, int, int)

def poly_overlaps (np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] query_boxes, np.int32_t device_id=0):
    assert boxes.shape[1] == 8 and query_boxes.shape[1] == 8
    cdef int N = boxes.shape[0]
    cdef int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype = np.float32)
    _poly_overlaps(&overlaps[0, 0], &boxes[0, 0], &query_boxes[0, 0], N, K, device_id)
    return overlaps

def poly_gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    if dets.shape[0] == 0:
        return []
    assert dets.shape[1] == 9

    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 8]
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] sorted_dets = dets[order, :]
    _poly_nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return list(order[keep])

