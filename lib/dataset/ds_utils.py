import numpy as np


def unique_boxes(boxes, scale=1.0):
    """ return indices of unique boxes """
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep

def area(boxes, num_of_inter):
    def trangle_area(a, b, c):
        return ((a[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1])
                - (a[:, 1] - c[:, 1]) * (b[:, 0] - c[:, 0])) / 2.0

    area = np.array([0.0] * boxes.shape[0])
    for i in range(num_of_inter - 2):
        area += abs(trangle_area(boxes[:, [0, 1]], boxes[:, [2 * i + 2, 2 * i + 3]],
                                 boxes[:, [2 * i + 4, 2 * i + 5]]))

    return area

def filter_small_quadrangle_boxes(boxes, min_size):
    """
    method 1: get outer bounding box
    method 2: calculate each side length
    for simplicity, we choose method 1
    :param boxes: numpy array of quadrangle boxes
    :param min_size:
    :return:
    """
    # x1, y1, x2, y2, x3, y3, x4, y4 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], \
    #                                  boxes[:, 4], boxes[:, 5], boxes[:, 6], boxes[:, 7]

    # xs = boxes[:, [0, 2, 4, 6]]
    # ys = boxes[:, [1, 3, 5, 7]]
    # xmin = xs.min(axis=1)
    # ymin = ys.min(axis=1)
    # xmax = xs.max(axis=1)
    # ymax = ys.max(axis=1)
    # ws = xmax - xmin + 1
    # hs = ymax - ymin + 1
    # keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    areas = area(boxes, num_of_inter=4)
    keep = np.where(areas >= min_size * min_size)[0]
    return keep

def get_horizon_minAreaRectangle(boxes):
    """
    get surrounding bbox
    :param boxes: numpy 8-num [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: 4-num [xmin, ymin, xmax, ymax]
    """
    assert boxes.shape[1] == 8, "boxes shape: {}".format(boxes.shape[1])
    xs = boxes[:, [0, 2, 4, 6]]
    ys = boxes[:, [1, 3, 5, 7]]
    xmin = xs.min(axis=1)
    ymin = ys.min(axis=1)
    xmax = xs.max(axis=1)
    ymax = ys.max(axis=1)
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()
    # we can also write as:
    # ex_x = np.vstack((gt_boxes[:, 0], gt_boxes[:, 2], gt_boxes[:, 4], gt_boxes[:, 6]))
    # ex_y = np.vstack((gt_boxes[:, 1], gt_boxes[:, 3], gt_boxes[:, 5], gt_boxes[:, 7]))
    # gt_boxes_bbox[:, 0] = np.amin(ex_x, axis=0)
    # gt_boxes_bbox[:, 1] = np.amin(ex_y, axis=0)
    # gt_boxes_bbox[:, 2] = np.amax(ex_x, axis=0)
    # gt_boxes_bbox[:, 3] = np.amax(ex_y, axis=0)

def get_best_begin_point(coordinate):
    '''
    choose the start point that closest to bounding box top-left corner
    :param coordinate: array of len 8
    :return:
    '''

    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)

    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
    distances = np.array([np.sum((coord - dst_coordinate) ** 2) for coord in combinate])
    sorted_idx = distances.argsort()
    return combinate[sorted_idx[0]].tolist()

def filter_clockwise_boxes(boxes):
    """
    Validates that the 4 points that a polygon are in clockwise order.
    :param boxes: numpy array cols-8
    :return: boxes indexes that satisfy clockwise order
    """

    assert boxes.shape[1] == 8, "Points list shape not valid: " + str(boxes.shape[1])

    edge = [
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] + boxes[:, 1]),
        (boxes[:, 4] - boxes[:, 2]) * (boxes[:, 5] + boxes[:, 3]),
        (boxes[:, 6] - boxes[:, 4]) * (boxes[:, 7] + boxes[:, 5]),
        (boxes[:, 0] - boxes[:, 6]) * (boxes[:, 1] + boxes[:, 7])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3]
    # keep = np.where(summatory <= 0)[0]
    # fyk: there are some boxes that are collinear, which area is 0, we filter those too small boxes.
    areas = area(boxes, num_of_inter=4)
    keep = np.where((summatory <= 0) & (areas >= 16))[0]
    return keep

if __name__ == '__main__':
    boxes = np.array([[0,1, 1,0, 2,1, 1,2],
                      [2,3, 4,5, 6,7, 1,9]
    ])
    print get_horizon_minAreaRectangle(boxes)