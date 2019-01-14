#encoding: utf-8
import cv2
import random
import numpy as np

def histEqualColor(img):
    """
    Histogram Equalization Of RGB Images
    :param img: cv2 image
    :return:
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    res = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return res

def random_swap(im):
    """随机变换通道"""
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    swap = perms[random.randrange(0, len(perms))]
    im = im[:, :, swap]
    return im

def random_bright(im, delta=32):
    delta = random.uniform(-delta, delta)
    im += delta
    im = im.clip(min=0, max=255)
    return im

def random_contrast(im, lower=0.5, upper=1.5):
    """随机变换对比度"""
    alpha = random.uniform(lower, upper)
    im *= alpha
    im = im.clip(min=0, max=255)
    return im

def _random_saturation(im, lower=0.5, upper=1.5):
    """随机变换饱和度"""
    im[:, :, 1] *= random.uniform(lower, upper)
    return im

def _random_hue(im, delta=18.0):
    """随机变换色度(HSV空间下(-180, 180))"""
    im[:, :, 0] += random.uniform(-delta, delta)
    im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
    im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
    return im

def apply_distort(im):
    """
    扭曲: 随机饱和度和色度变换
    变换HSV空间，然后变回到OpenCV的BGR空间
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = im.astype(np.float)
    im = _random_saturation(im)
    im = _random_hue(im)
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    return im


def salt_and_pepper(img, prob=.005):
    """
    Applies salt and pepper noise to an image with given probability for both.
    Args:
        img: the image to be augmented in array format
        prob: the probability of applying noise to the image
    Output:
        Augmented image
    """

    newimg = np.copy(img)
    whitemask = np.random.randint(0, int((1 - prob) * 200), size=img.shape[:2])
    blackmask = np.random.randint(0, int((1 - prob) * 200), size=img.shape[:2])
    newimg[whitemask == 0] = 255
    newimg[blackmask == 0] = 0

    return newimg


def tmp_distort(im):
    if random.random() < 0.5:
        random_bright(im)
    if random.random() < 0.5:
        random_contrast(im)
    if random.random() < 0.5:
        apply_distort(im)
    if random.random() < 0.1:
        random_swap(im)