# --------------------------------------------------------
# Rotate RPN - proposal with quadrangle
# Licensed under The MIT License [see LICENSE for details]
# Modified by fyk
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------
"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from bbox.rbbox_transform import bbox_pred_quadrangle, clip_quadrangle_boxes
from rpn.generate_rotate_anchor import generate_rotate_anchors
# from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.nms_poly import *

from dataset.ds_utils import *
from utils.dplog import Logger as logger
from utils.tictoc import *
DEBUG = False


class ProposalOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios, angles, output_score, output_horizon_rois,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size):
        super(ProposalOperator, self).__init__()
        self._feat_stride = feat_stride
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._angles = np.fromstring(angles[1:-1], dtype=float, sep=',')
        self._anchors = generate_rotate_anchors(base_size=self._feat_stride, scales=self._scales, ratios=self._ratios, angles=self._angles)
        self._num_anchors = self._anchors.shape[0]
        self._box_dim = self._anchors.shape[1]
        assert self._box_dim == 8 # quadrangle
        self._output_score = output_score
        self._output_horizon_rois = output_horizon_rois
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self._threshold = threshold
        self._rpn_min_size = rpn_min_size

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

    def forward(self, is_train, req, in_data, out_data, aux):
        # nms = gpu_nms_wrapper(self._threshold, in_data[0].context.device_id)
        # nms = cpu_nms_poly_wrapper(self._threshold)
        # nms = gpu_nms_poly_wrapper(self._threshold, in_data[0].context.device_id)
        nms_r = gpu_nms_poly_wrapper_r(self._threshold, in_data[0].context.device_id)

        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        pre_nms_topN = self._rpn_pre_nms_top_n
        post_nms_topN = self._rpn_post_nms_top_n
        min_size = self._rpn_min_size

        # the first set of anchors are background probabilities
        # keep the second part
        scores = in_data[0].asnumpy()[:, self._num_anchors:, :, :]
        bbox_deltas = in_data[1].asnumpy()
        im_info = in_data[2].asnumpy()[0, :]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox_deltas and shifted anchors
        # use real image size instead of padded feature map sizes
        height, width = int(im_info[0] / self._feat_stride), int(im_info[1] / self._feat_stride)

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)
            print "resudial: {}".format((scores.shape[2] - height, scores.shape[3] - width))

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, self._box_dim)) + shifts.reshape((1, K, self._box_dim)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, self._box_dim))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, self._box_dim))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = self._clip_pad(scores, (height, width))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_pred_quadrangle(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_quadrangle_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = self._filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # fyk: remove non clockwise order boxes (some boxes are not normal quandrangle)
        keep = filter_clockwise_boxes(proposals)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        det = np.hstack((proposals, scores)).astype(np.float32)
        # logger.debug("start poly nms")
        # tic()
        # keep = nms(det)
        keep = nms_r(det)
        # compare
        # logger.debug("gpu_nms_poly cost {} on device {}".format(toc(), in_data[0].context.device_id))
        # tic()
        # keep_r = nms_r(det)
        # logger.debug("gpu_nms_poly_r cost {} on device {}".format(toc(), in_data[0].context.device_id))
        # if keep_r != keep:
        #     logger.info("gpu_nms_poly: {}".format(len(keep)))
        #     logger.info("gpu_nms_poly_r: {}".format(len(keep_r)))
        # else:
        #     logger.info("nms same!")

        # logger.debug("pse start poly nms")
        # keep = [1] * det.shape[0]
        # logger.debug("end poly nms")
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_topN:
            pad = npr.choice(keep, size=post_nms_topN - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)

        out_idx = 0
        if self._output_horizon_rois:
            # fyk: get bbox of quadrangle rois: [batch_idx, x1, y1, x2, y2, x3, y3, x4, y4]
            bbox_proposals = get_horizon_minAreaRectangle(proposals)
            bbox_proposals = np.hstack((batch_inds, bbox_proposals))
            out_idx += 1
            self.assign(out_data[out_idx], req[out_idx], bbox_proposals.astype(np.float32, copy=False))

        if self._output_score:
            out_idx += 1
            self.assign(out_data[out_idx], req[out_idx], scores.astype(np.float32, copy=False))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        return filter_small_quadrangle_boxes(boxes, min_size)

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


@mx.operator.register("proposal_rotate")
class ProposalProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='16', scales='(8, 16, 32)', ratios='(0.5, 1, 2)',
                 angles='(-60, -30, 0, 30, 60, 90)',
                 output_score='False', output_horizon_rois='False',
                 rpn_pre_nms_top_n='6000', rpn_post_nms_top_n='300', threshold='0.3', rpn_min_size='16'):
        super(ProposalProp, self).__init__(need_top_grad=False)
        self._feat_stride = int(feat_stride)
        self._scales = scales
        self._ratios = ratios
        self._angles = angles
        self._output_score = strtobool(output_score)
        self._output_horizon_rois = strtobool(output_horizon_rois)
        self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self._threshold = float(threshold)
        self._rpn_min_size = int(rpn_min_size)

    def list_arguments(self):
        return ['cls_prob', 'bbox_pred', 'im_info']

    def list_outputs(self):
        out_blob = ['output']
        if self._output_horizon_rois:
            out_blob.append('horizon_rois')
        if self._output_score:
            out_blob.append('score')
        return out_blob

    def infer_shape(self, in_shape):
        cls_prob_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        assert cls_prob_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in cls and reg'

        batch_size = cls_prob_shape[0]
        im_info_shape = (batch_size, 3)
        box_dim = 8
        output_shape = (self._rpn_post_nms_top_n, box_dim + 1)
        output_horizon_shape  = (self._rpn_post_nms_top_n, 4 + 1)
        score_shape = (self._rpn_post_nms_top_n, 1)

        out_shape = [output_shape]
        if self._output_horizon_rois:
            out_shape.append(output_horizon_shape)
        if self._output_score:
            out_shape.append(score_shape)

        return [cls_prob_shape, bbox_pred_shape, im_info_shape], out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalOperator(self._feat_stride, self._scales, self._ratios, self._angles, self._output_score,
                                self._output_horizon_rois,
                                self._rpn_pre_nms_top_n, self._rpn_post_nms_top_n, self._threshold, self._rpn_min_size)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
