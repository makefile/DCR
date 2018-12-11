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
    xs = boxes[:, [0, 2, 4, 6]]
    ys = boxes[:, [1, 3, 5, 7]]
    xmin = xs.min(axis=1)
    ymin = ys.min(axis=1)
    xmax = xs.max(axis=1)
    ymax = ys.max(axis=1)
    ws = xmax - xmin + 1
    hs = ymax - ymin + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def get_horizen_minAreaRectangle(boxes):
    """
    get surrounding bbox
    :param boxes: numpy 8-num [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: 4-num [xmin, ymin, xmax, ymax]
    """
    assert boxes.shape[1] == 8
    xs = boxes[:, [0, 2, 4, 6]]
    ys = boxes[:, [1, 3, 5, 7]]
    xmin = xs.min(axis=1)
    ymin = ys.min(axis=1)
    xmax = xs.max(axis=1)
    ymax = ys.max(axis=1)
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()

def get_best_begin_point(coordinate):
    '''
    choose the start point that closest to bounding box top-left corner
    :param coordinate:
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
