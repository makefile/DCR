import numpy as np
# print all elements of numpy array
np.set_printoptions(threshold=np.nan)

def in_rect( pt_x,  pt_y, pts) :
    # // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
    # print type(pts)
    isInside = False;
    # // the previous bounding box check can remove false-negatives in edge-cases
    # // remove the previous check code to speed up if you don't care of edge-cases
    n = 4 # point num
    j = n - 1
    for i in range(n):

        ix = pts[i * 2]
        iy = pts[i * 2 + 1]
        jx = pts[j * 2]
        jy = pts[j * 2 + 1]
        if ( (iy > pt_y) != (jy > pt_y) and
                pt_x < (jx - ix) * (pt_y - iy) / (jy - iy) + ix ):
            isInside = not isInside

        j = i; i+=1

    return isInside

def test_in_rect():
    pts = [10.2, 5.2, 55.2, 5.2, 55.2, 45.2, 10.2, 45.2]
    cnt = 0
    for i in range(100):
        for j in range(100):
            if in_rect(i, j, pts):
                cnt += 1
    print cnt

def test_fill_poly():
    import cv2

    a = np.zeros((10, 10))
    area = np.array([[1, 3], [4, 3], [4, 8], [1, 9]], np.int32)
    # faster than fillPoly (which is both for convex poly and complex poly)
    cv2.fillConvexPoly(a, area, 1)
    area = np.array([[3, 1], [8, 1], [8, 5], [3, 6]], np.int32)
    cv2.fillConvexPoly(a, area, 1)
    print a

def test_fill_poly2():
    import cv2
    bin_mask_gt = np.zeros((35, 35), np.uint8)
    gt_boxes = np.array([[0, 12.0, 12.60, 7.40, 0.10, 29.70, 0.10, 31.30, 3.60]])
    areas = []
    # [batch_idx, x1, y1, x2, y2, x3, y3, x4, y4] point in clockwise
    for b in gt_boxes:
        a = np.array([ [round(b[1]), round(b[2])], [round(b[3]), round(b[4])],
                       [round(b[5]), round(b[6])], [round(b[7]), round(b[8])] ],
                     np.int32) # numpy type for fillPoly must be integer
        areas.append(a)

    cv2.fillPoly(bin_mask_gt, areas, color=1)
    print bin_mask_gt

def test_broadcast_mul():
    import numpy as np
    import mxnet as mx
    a = mx.nd.array(np.array([1, 2]))
    b = mx.nd.array(np.array([[2], [4]]))
    o = mx.nd.broadcast_mul(a, b).asnumpy()
    print o
    # [N, C, H, W] x [N, 1, H, W]
    a = mx.nd.array(np.array([[ [[1, 2],
                                 [3, 4]
                                ],
                               [[5, 6],
                                [7, 8]
                                ]]]))
    b = mx.nd.array(np.array([[ [[0.1, 0.2],
                                 [0.3, 0.4]
                                ] ]]))

    o = mx.nd.broadcast_mul(a, b)
    print o.asnumpy()
    part = mx.nd.slice_axis(o, axis=1, begin=1, end=2)
    print part.asnumpy()

if __name__ == '__main__':
    test_fill_poly()
    test_fill_poly2()

