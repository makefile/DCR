# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import cPickle
import os, sys
import time
import mxnet as mx
import numpy as np

from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from bbox.rbbox_transform import bbox_pred_quadrangle, clip_quadrangle_boxes
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.polygon_nms import py_polygon_nms_wrapper
from utils.PrefetchingIter import PrefetchingIter
import cv2

class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def im_proposal(predictor, data_batch, data_names, scales):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    boxes_all = []

    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        # transform to original scale
        boxes = boxes / scale
        scores_all.append(scores)
        boxes_all.append(boxes)

    return scores_all, boxes_all, data_dict_all


def generate_proposals(predictor, test_data, imdb, cfg, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    idx = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_proposal(predictor, data_batch, data_names, scales)
        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict, scale) in enumerate(zip(scores_all, boxes_all, data_dict_all, scales)):
            # assemble proposals
            dets = np.hstack((boxes, scores))
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4:] > thresh)[0]
            dets = dets[keep, :]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale, cfg)

            print 'generating %d/%d' % (idx + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
                'data %.4fs net %.4fs' % (t1, t2 / test_data.batch_size)
            idx += 1


    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.result_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


def im_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
    return scores_all, pred_boxes_all, data_dict_all

# TODO for quadrangle test
def im_detect_quadrangle(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        # post processing
        pred_boxes = bbox_pred_quadrangle(rois, bbox_deltas)
        pred_boxes = clip_quadrangle_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
    return scores_all, pred_boxes_all, data_dict_all


# TODO for quadrangle test
def pred_eval_quadrangle(predictor, test_data, imdb, cfg, vis=False, draw=False,
                          thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        # imdb.count_ar()
        imdb.check_transform()
        imdb.draw_gt_and_detections(all_boxes, thresh=0.1)
        info_str = imdb.evaluate_detections(all_boxes)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 9 array of detections in
    #    (x1, y1, x2, y2, x3, y3, x4, y4, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all= im_detect_quadrangle(predictor, data_batch, data_names, scales, cfg)

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            # idx = int(data_dict['im_index'])-1
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 8:16] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 8:(j + 1) * 8]
                temp_cls_boxes = np.zeros((cls_boxes.shape[0], 4), dtype=cls_boxes.dtype)
                temp_cls_boxes_x = np.vstack((cls_boxes[:, 0], cls_boxes[:, 2], cls_boxes[:, 4], cls_boxes[:, 6]))
                temp_cls_boxes_y = np.vstack((cls_boxes[:, 1], cls_boxes[:, 3], cls_boxes[:, 5], cls_boxes[:, 7]))
                temp_cls_boxes[:, 0] = np.amin(temp_cls_boxes_x, axis=0)
                temp_cls_boxes[:, 1] = np.amin(temp_cls_boxes_y, axis=0)
                temp_cls_boxes[:, 2] = np.amax(temp_cls_boxes_x, axis=0)
                temp_cls_boxes[:, 3] = np.amax(temp_cls_boxes_y, axis=0)
                cls_dets = np.hstack((temp_cls_boxes, cls_scores))
                cls_quadrangle_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j][idx+delta] = cls_quadrangle_dets[keep, :]

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
                vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)

            if draw:
                if not os.path.isdir(cfg.TEST.save_img_path):
                    os.mkdir(cfg.TEST.save_img_path)
                path = os.path.join(cfg.TEST.save_img_path, str(idx) + '.jpg')
                boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, imdb.num_classes)]
                im = draw_all_quadrangle_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg, threshold=0.2)
                print path
                cv2.imwrite(path, im)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
        if logger:
            logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # imdb.draw_gt_and_detections(all_boxes, thresh=0.1)
    info_str = imdb.evaluate_detections(all_boxes)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))

# TODO for dota test
def pred_eval_dota_quadrangle(predictor, test_data, imdb, cfg, vis=False, draw=False, draw_gt=False,
                          thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    # ignore_cache = True
    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        # imdb.count_ar()
        #imdb.check_transform()
        # imdb.draw_gt_and_detections(all_boxes, thresh=0.1)
        info_str = imdb.evaluate_detections(all_boxes)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    #nms = py_nms_wrapper(cfg.TEST.NMS)
    nms = py_polygon_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 9 array of detections in
    #    (x1, y1, x2, y2, x3, y3, x4, y4, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    all_boxes_vis = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all= im_detect_quadrangle(predictor, data_batch, data_names, scales, cfg)

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            # idx = int(data_dict['im_index'])-1
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 8:16] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 8:(j + 1) * 8]
                cls_quadrangle_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_quadrangle_dets)
                all_boxes_vis[j][idx + delta] = cls_quadrangle_dets[keep, :]
                all_boxes[j][idx+delta] = cls_quadrangle_dets

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes_vis[j][idx+delta] for j in range(1, imdb.num_classes)]
                vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)

            if draw:
                if not os.path.isdir(cfg.TEST.save_img_path):
                    os.mkdir(cfg.TEST.save_img_path)
                # path = os.path.join(cfg.TEST.save_img_path, str(idx + delta) + '.jpg')
                img_set_idx = imdb.image_set_index[idx + delta]
                path = os.path.join(cfg.TEST.save_img_path, img_set_idx + '.jpg')
                boxes_this_image = [[]] + [all_boxes_vis[j][idx + delta] for j in range(1, imdb.num_classes)]
                im = draw_all_quadrangle_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg, threshold=0.2)

                if draw_gt: # fyk: draw GT boxes too.
                    gt_db = imdb.load_bboxes(img_set_idx)
                    # print img_set_idx, gt_db
                    gt_boxes = gt_db['boxes']
                    draw_gt_boxes(im, gt_boxes)

                print('draw det image to {}'.format(path))
                cv2.imwrite(path, im)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3
        info_str = 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
        sys.stdout.write('\r' + info_str)
        sys.stdout.flush()
        if logger:
            logger.info(info_str)

    sys.stdout.write('\n')
    sys.stdout.flush()

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # imdb.draw_gt_and_detections(all_boxes, thresh=0.1)
    # info_str = imdb.evaluate_detections(all_boxes)
    info_str = ''
    imdb.write_results_by_class(all_boxes, threshold=0.0)

    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))

def pred_eval_quadrangle_multiscale(prefix, predictor, test_data, imdb, cfg, vis=False, draw=False,
                          thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    det_file = os.path.join(imdb.result_path, imdb.name + '{}'.format(prefix) + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    # nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 9 array of detections in
    #    (x1, y1, x2, y2, x3, y3, x4, y4, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all= im_detect_quadrangle(predictor, data_batch, data_names, scales, cfg)

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            # idx = int(data_dict['im_index'])-1
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 8:16] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 8:(j + 1) * 8]
                # temp_cls_boxes = np.zeros((cls_boxes.shape[0], 4), dtype=cls_boxes.dtype)
                # temp_cls_boxes_x = np.vstack((cls_boxes[:, 0], cls_boxes[:, 2], cls_boxes[:, 4], cls_boxes[:, 6]))
                # temp_cls_boxes_y = np.vstack((cls_boxes[:, 1], cls_boxes[:, 3], cls_boxes[:, 5], cls_boxes[:, 7]))
                # temp_cls_boxes[:, 0] = np.amin(temp_cls_boxes_x, axis=0)
                # temp_cls_boxes[:, 1] = np.amin(temp_cls_boxes_y, axis=0)
                # temp_cls_boxes[:, 2] = np.amax(temp_cls_boxes_x, axis=0)
                # temp_cls_boxes[:, 3] = np.amax(temp_cls_boxes_y, axis=0)
                # cls_dets = np.hstack((temp_cls_boxes, cls_scores))
                cls_quadrangle_dets = np.hstack((cls_boxes, cls_scores))
                # keep = nms(cls_dets)
                all_boxes[j][idx+delta] = cls_quadrangle_dets

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

            # if vis:
            #     boxes_this_image = [[]] + [all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
            #     vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)
            #
            if draw:
                if not os.path.isdir(cfg.TEST.save_img_path):
                    os.mkdir(cfg.TEST.save_img_path)
                # path = os.path.join(cfg.TEST.save_img_path, '{}_'.format(prefix) + str(idx + delta) + '.jpg')
                img_set_idx = imdb.image_set_index[idx + delta]
                path = os.path.join(cfg.TEST.save_img_path, '{}_'.format(prefix) + img_set_idx + '.jpg')
                boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, imdb.num_classes)]
                im = draw_all_quadrangle_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg, threshold=0.2)
                print path
                cv2.imwrite(path, im)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
        if logger:
            logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))

    threshold = 0.1
    detection_results_path = os.path.join(imdb.result_path, 'test_{}_results'.format(prefix))
    if not os.path.isdir(detection_results_path):
        os.mkdir(detection_results_path)
    for cls_ind, cls in enumerate(imdb.classes):
        if cls == '__background__':
            continue
        for im_ind, index in enumerate(imdb.image_set_index):
            dets = all_boxes[cls_ind][im_ind]
            with open(os.path.join(imdb.result_path, 'test_{}_results'.format(prefix), 'res_img_{}.txt'.format(index)), 'wt') as f:
                for k in range(dets.shape[0]):
                    if dets[k, 8] <= threshold:
                        continue
                    if imdb.validate_clockwise_points(dets[k, 0:8]):
                        f.write(
                            '{},{},{},{},{},{},{},{},{}\n'.format(int(dets[k, 0]), int(dets[k, 1]), int(dets[k, 2]),
                                                               int(dets[k, 3]),
                                                               int(dets[k, 4]), int(dets[k, 5]), int(dets[k, 6]),
                                                               int(dets[k, 7]), dets[k, 8]))
                    else:
                        print 'A detected box is anti-clockwise! Index:{}'.format(index)
                        print dets[k, 0:8]

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

"""
pred for dota dataset, ignore_cache and don't use NMS
"""
def pred_eval_dota(predictor, test_data, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    ignore_cache = True
    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        info_str = imdb.evaluate_detections(all_boxes)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    # nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    # print len(all_boxes)
    # print len(all_boxes[0])
    # print len(all_boxes[0][0])

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_detect(predictor, data_batch, data_names, scales, cfg)

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                # keep = nms(cls_dets)
                print j,idx,delta
                print cls_dets.shape
                all_boxes[j][idx+delta] = cls_dets

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
                vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
        if logger:
            logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_detections(all_boxes)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))



def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-3):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def draw_all_quadrangle_detection(im_array, detections, class_names, scale, cfg, threshold=0.2):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2, random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:8] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            for i in range(3):
                cv2.line(im, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2)
            cv2.line(im, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
            cv2.putText(im, '%s %.2f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def draw_gt_boxes(img, gt_boxes):
    import cv2

    color = (0, 255, 0) # GT box draw in green color
    img_height, img_width = img.shape[0], img.shape[1]
    for k in range(gt_boxes.shape[0]):
        bbox = gt_boxes[k, :8]
        bbox = map(int, bbox)

        xmax = max(bbox[0], bbox[2], bbox[4], bbox[6])
        ymax = max(bbox[1], bbox[3], bbox[5], bbox[7])
        if xmax > img_width:
            print "extreme xmax", xmax
        if ymax > img_height:
            print "extreme ymax", ymax
        for i in range(3):
            cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]),
                     color=color, thickness=1)
        cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=1)

