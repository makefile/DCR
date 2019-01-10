from cpu_nms_poly import poly_overlaps_cython, cpu_nms_poly
from gpu_nms_poly import poly_overlaps, poly_gpu_nms
from rbbox_overlaps import rbbox_overlaps, rotate_gpu_nms

def cpu_nms_poly_wrapper(thresh):
    def _nms(dets):
        return cpu_nms_poly(dets, thresh)
    return _nms

def gpu_nms_poly_wrapper(thresh, device_id):
    def _nms(dets):
        return poly_gpu_nms(dets, thresh, device_id)
    return _nms

def cpu_polygon_overlaps(boxes, query_boxes):
    return poly_overlaps_cython(boxes, query_boxes)

def gpu_polygon_overlaps(boxes, query_boxes, device_id):
    return poly_overlaps(boxes, query_boxes, device_id)

def gpu_polygon_overlaps_r(boxes, query_boxes, device_id):
    return rbbox_overlaps(boxes, query_boxes, device_id)

def gpu_nms_poly_wrapper_r(thresh, device_id):
    def _nms(dets):
        return rotate_gpu_nms(dets, thresh, device_id)
    return _nms

if __name__ == '__main__':
    import numpy as np
    from dataset.ds_utils import filter_clockwise_boxes
    '''
    # float num clockwise
    boxes = np.array([[0, 0, 1, 0, 1, 1, 0, 1]], dtype=np.float32)
    query_boxes = np.array([[0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5]], dtype=np.float32)
    # IOU should be 0.142857
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
    print '-' * 50
    # int num
    boxes = np.array([[0, 0, 2, 0, 2, 2, 0, 2]], dtype=np.float32)
    query_boxes = np.array([[1, 1, 3, 1, 3, 3, 1, 3]], dtype=np.float32)
    # IOU should be 0.142857
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
    print '-' * 50
    # float num boxes: anti-clockwise
    boxes = np.array([[0, 1, 1, 1, 1, 0, 0, 0]], dtype=np.float32)
    query_boxes = np.array([[0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    # IOU should be 0.142857
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
    print '-' * 50
    # random order
    boxes = np.array([[0, 1, 1, 0, 1, 1, 0, 0]], dtype=np.float32)
    query_boxes = np.array([[0.5, 1.5, 1.5, 1.5, 0.5, 0.5, 1.5, 0.5]], dtype=np.float32)
    keep = filter_clockwise_boxes(query_boxes[:, :8])
    print 'filtered box: ', len(keep)
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
    print '-' * 50
    dets = np.array([
        [0, 100, 0, 0, 100, 0, 100, 100, 0.99],
        [10, 110, 10, 10, 130, 10, 150, 110, 0.88],
        [150, 250, 150, 150, 250, 150, 250, 250, 0.77],
        [20, 50, 50, 20, 120, 50, 50, 120, 0.66],
    ], dtype=np.float32)
    print cpu_nms_poly(dets, thresh=0.3)
    print poly_gpu_nms(dets, thresh=0.3, device_id=0)
    print '-' * 50
    boxes = np.array([[10.2,5.2, 55.2,5.2, 55.2,45.2, 10.2,45.2]], dtype=np.float32)
    query_boxes = np.array([[5.2,10.2, 50.2,5.2, 53.2,25.2, 7.2,40.2]], dtype=np.float32)
    # IOU should be 0.142857
    print poly_overlaps_cython(boxes, query_boxes)
    print gpu_polygon_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
    print '-' * 50
    dets = np.array([[10.2,5.2, 55.2,5.2, 55.2,45.2, 10.2,45.2, 0.99],
                     [5.2, 10.2, 50.2, 5.2, 53.2, 25.2, 7.2, 40.2, 0.88]], dtype=np.float32)
    print cpu_nms_poly(dets, thresh=0.3)
    print poly_gpu_nms(dets, thresh=0.3, device_id=0)
    print '-' * 50
    dets = np.array([[1.28059536e+02, 7.53659614e+02, 1.54156870e+02, 7.31694809e+02,
                      2.10754025e+02, 8.41808661e+02, 1.84801604e+02, 8.66288305e+02,
                            6.75240811e-03],
                     [1.19157898e+02, 7.46695847e+02, 1.71674001e+02, 7.24679275e+02,
                      2.13961131e+02, 8.45522432e+02, 1.66759067e+02, 8.73019281e+02,
                            4.05927701e-03],
                     [1.18819201e+02, 7.47947030e+02, 1.67174185e+02, 7.27417076e+02,
                      2.10358292e+02, 8.35907683e+02, 1.67939997e+02, 8.57752109e+02,
                            1.28786594e-01]], dtype=np.float32)

    print cpu_nms_poly(dets, thresh=0.4)
    print poly_gpu_nms(dets, thresh=0.4, device_id=0)

    print '-' * 50
    # special case that the boxes is just a loop line
    boxes = np.array([[6.70500000e+03, 7.98000000e+02, 6.72500000e+03, 7.98000000e+02,
                      6.72500000e+03, 7.98000000e+02, 6.70500000e+03, 7.98000000e+02]], dtype=np.float32)
    query_boxes = np.array([[6.71000000e+03, 7.92000000e+02, 6.72900000e+03, 7.92000000e+02,
                            6.72900000e+03, 7.98000000e+02, 6.71000000e+03, 7.98000000e+02]], dtype=np.float32)
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)

    print '-' * 50
    boxes = np.array([[576.1,576.1,686.9,512.1,750.9,622.9,640.1,686.9]], dtype=np.float32)
    query_boxes = np.array([[792.0,746.0,799.0,746.0,799.0,799.0,792.0,799.0]], dtype=np.float32)
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
    '''
    boxes = np.array([[4.4750000e+03, 2.2980000e+03, 4.6480000e+03, 2.2980000e+03, 4.6490000e+03,
                        2.2980000e+03, 4.4870000e+03, 2.2980000e+03, 3.2260695e-03]], dtype=np.float32)
    query_boxes = np.array([[4.44500000e+03, 2.29800000e+03, 4.62000000e+03, 2.29800000e+03,
                             4.62100000e+03, 2.29800000e+03, 4.45500000e+03, 2.29800000e+03,
                             2.80404254e-03]], dtype=np.float32)
    # todo check IOU between [0,1], any bad value or warning would cause problem
    print poly_overlaps_cython(boxes[:, :8], query_boxes[:, :8])
    print poly_overlaps(boxes[:, :8], query_boxes[:, :8], device_id=0)
    print rbbox_overlaps(boxes[:, :8], query_boxes[:, :8], device_id=0)
    # the box is a line [5400, 3024, 5400, 3029, 5400, 3076, 5400, 3076]
    boxes = np.array([[5.40000000e+03, 3.02400000e+03, 5.40000000e+03, 3.02900000e+03,
                       5.40000000e+03, 3.07600000e+03, 5.40000000e+03, 3.07600000e+03, 2.32902286e-03]], dtype=np.float32)
    query_boxes = boxes
    keep = filter_clockwise_boxes(query_boxes[:, :8])
    print poly_overlaps_cython(boxes[:, :8], query_boxes[:, :8])
    print poly_overlaps(boxes[:, :8], query_boxes[:, :8], device_id=0)
    print rbbox_overlaps(boxes[:, :8], query_boxes[:, :8], device_id=0)
    # test same box IoU=0
    boxes = np.array([[576.1, 576.1, 686.9, 512.1, 750.9, 622.9, 640.1, 686.9]], dtype=np.float32)
    query_boxes = boxes
    print poly_overlaps_cython(boxes, query_boxes)
    print poly_overlaps(boxes, query_boxes, device_id=0)
    print rbbox_overlaps(boxes, query_boxes, device_id=0)
