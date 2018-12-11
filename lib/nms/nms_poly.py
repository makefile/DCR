from cpu_nms_poly import poly_overlaps_cython, cpu_nms_poly
from gpu_nms_poly import rotate_gpu_nms
from rbbox_overlaps import rbbox_overlaps

def cpu_nms_poly_wrapper(thresh):
    def _nms(dets):
        return cpu_nms_poly(dets, thresh)
    return _nms

def gpu_nms_poly_wrapper(thresh, device_id):
    def _nms(dets):
        return rotate_gpu_nms(dets, thresh, device_id)
    return _nms

def cpu_polygon_overlaps(boxes, query_boxes):
    return poly_overlaps_cython(boxes, query_boxes)

def gpu_polygon_overlaps(boxes, query_boxes, device_id):
    return rbbox_overlaps(boxes, query_boxes, device_id)
