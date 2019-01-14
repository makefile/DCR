# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by fyk
# --------------------------------------------------------

import numpy as np
import os
import cv2
import random
from math import ceil
from image import resize, transform
from bbox.rbbox_transform import clip_quadrangle_boxes, clip_rotate_boxes
from image_augmenter import histEqualColor, apply_distort

def get_image_quadrangle_bboxes(roidb, config, isTrain=True):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if isTrain:
            if roidb[i]['flipped']: im = im[:, ::-1, :]
            # do hue/saturation distort,
            # do not do brightness and contrast distort since we always do Histogram Equalization
            if random.random() < 0.5:
                im = apply_distort(im)
        # Histogram Equalization
        im = histEqualColor(im)

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        # fyk align image for shape match in upsample layer
        do_size_align = False
        if do_size_align:
            im_size_align = 16
            new_im_height = int(ceil(im_tensor.shape[2] / float(im_size_align))) * im_size_align
            new_im_width  = int(ceil(im_tensor.shape[3] / float(im_size_align))) * im_size_align
            padded_im = np.zeros((1, 3, new_im_height, new_im_width), dtype=im_tensor.dtype)
            padded_im[:, :, :im_tensor.shape[2], :im_tensor.shape[3]] = im_tensor
            im_tensor = padded_im
        processed_ims.append(im_tensor)

        if isTrain:
            if config.box_type == config.BOX_TYPE_QUADRANGLE:
                new_rec['boxes'] = clip_quadrangle_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
            else: # BOX_TYPE_ROTATE_RECTANGLE
                new_rec['boxes_rotate'] = clip_rotate_boxes(np.round(roi_rec['boxes_rotate'].copy() * im_scale), im_info[:2])

        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def get_image_test(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        # if roidb[i]['flipped']:
        #     im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        # new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def resize_to_fix_size(im, target_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    # im_scale = [height_scale, width_scale]
    im_scale = [float(target_size[0])/ float(im_shape[0]), float(target_size[1])/ float(im_shape[1])]
    im = cv2.resize(im, None, None, fx=im_scale[1], fy=im_scale[0], interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

