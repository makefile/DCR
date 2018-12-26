import numpy as np
from math import pi, atan2


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

def is_quadrangle_simple(box):
    '''
    simple means there is no intersections between side lines.
    compare each segment against all others and check for intersections. Complexity O(n^2).
    :return:
    '''

    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    # // Given three colinear points p, q, r, the function checks if
    # // point q lies on line segment 'pr'
    def onSegment(px, py, qx, qy, rx, ry):
        if (qx <= max(px, rx) and qx >= min(px, rx) and
                qy <= max(py, ry) and qy >= min(py, ry)):
            return True

        return False

    # // To find orientation of ordered triplet (p, q, r).
    # // The function returns following values
    # // 0 --> p, q and r are colinear
    # // 1 --> Clockwise
    # // 2 --> Counterclockwise
    def orientation(px, py, qx, qy, rx, ry):
        # // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
        # // for details of below formula.
        val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if val == 0:
            return 0  # colinear
        elif val > 0:
            return 1  # clock
        else:
            return 2  # counterclock wise

    # // The main function that returns true if line segment 'p1q1'
    # // and 'p2q2' intersect.
    def doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):

        # // Find the four orientations needed for general and
        # // special cases
        o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y)
        o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y)
        o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y)
        o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y)

        # // General case
        if o1 != o2 and o3 != o4: return True

        # // Special Cases, p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if o1 == 0 and onSegment(p1x, p1y, p2x, p2y, q1x, q1y): return True
        if o2 == 0 and onSegment(p1x, p1y, q2x, q2y, q1x, q1y): return True
        if o3 == 0 and onSegment(p2x, p2y, p1x, p1y, q2x, q2y): return True
        if o4 == 0 and onSegment(p2x, p2y, q1x, q1y, q2x, q2y): return True

        return False  # Doesn't fall in any of the above cases

    def intersects(s, seg):
        # check if intersect
        if doIntersect(s[0], s[1], s[2], s[3], seg[0], seg[1], seg[2], seg[3]):
            return True
        return False

    # x1, y1, x2, y2, x3, y3, x4, y4 = box
    # four segment lines
    segs = []
    for i in range(4):
        segs.append((box[i * 2], box[i * 2 + 1], box[(i * 2 + 2) % 8], box[(i * 2 + 30) % 8]))
    # only consider opposite seg line
    if intersects(segs[0], segs[2]) or intersects(segs[1], segs[3]): return False

    return True


TWO_PI = 2 * pi
# https://stackoverflow.com/questions/471962/how-do-i-efficiently-determine-if-a-polygon-is-convex-non-convex-or-complex
def is_convex_polygon(polygon):
    """
    Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.

    :param polygon: [(x1,y1),...,(xn,yn)]
    :return: boolean
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon

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
    # fyk: there are some boxes that are collinear(whose area is 0), also we filter those too small boxes.
    areas = area(boxes, num_of_inter=4)
    # boxes_is_simple = np.array([is_quadrangle_simple(b) for b in boxes])
    # keep = np.where((summatory <= 0) & (boxes_is_simple == True))[0]
    # filter out non-convex box & non-simple polygon
    boxes_is_simple_convex = np.array([is_convex_polygon([(b[0],b[1]),(b[2],b[3]),(b[4],b[5]),(b[6],b[7])])
                      for b in boxes])
    keep = np.where((summatory <= 0) & (areas >= 16) & (boxes_is_simple_convex == True))[0]

    return keep

if __name__ == '__main__':
    boxes = np.array([[0,1, 1,0, 2,1, 1,2],
                      [2,3, 4,5, 6,7, 1,9]
    ])
    print get_horizon_minAreaRectangle(boxes)