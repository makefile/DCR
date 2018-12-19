import numpy as np

from cpu_nms import cpu_nms
from gpu_nms import gpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def py_softnms_wrapper(thresh, max_dets=-1):
    def _nms(dets):
        return soft_nms(dets, thresh, max_dets)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores


def soft_nms(dets, thresh, max_dets):
    if dets.shape[0] == 0:
        return np.zeros((0, 5))

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    if max_dets == -1:
        max_dets = order.size

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0

    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]
        dets[i, 4] = scores[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = rescore(ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]
    dets = dets[keep, :]
    return dets


if __name__ == '__main__':

    dets = np.array([[1.54879288e+02, 5.63751030e+01, 2.00795334e+02, 1.76900604e+02, 9.13123965e-01],
                     [1.54429581e+02, 5.55528603e+01, 2.00706253e+02, 1.78303726e+02, 9.10487592e-01],
                     [8.22127075e+01, 6.74252090e+01, 1.21588181e+02, 1.85179321e+02, 8.45470846e-01],
                     [8.26999969e+01, 6.73755112e+01, 1.21075806e+02, 1.85062042e+02, 8.18630576e-01],
                     [1.52734726e+02, 5.55439453e+01, 1.99126022e+02, 1.76959335e+02, 5.77099562e-01],
                     [2.87755089e+01, 6.29712677e+01, 9.89483795e+01, 2.14977646e+02, 4.13117468e-01],
                     [4.41039162e+01, 6.47515717e+01, 9.48828735e+01, 2.05074326e+02, 2.21668169e-01],
                     [2.20986038e+02, 5.58766098e+01, 2.48486771e+02, 1.75349884e+02, 1.01995334e-01]],
                    dtype=np.float32)

    print nms(dets, thresh=0.3)
    print gpu_nms(dets, thresh=0.3, device_id=0)
