# --------------------------------------------------------
# Rotate RPN - real quadrangle proposal for training
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
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import cPickle
from distutils.util import strtobool

from core.rcnn_quadrangle import sample_rois_rotate
from dataset.ds_utils import get_horizon_minAreaRectangle

DEBUG = False


class ProposalTargetQuadrangleOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction, output_horizon_rois):
        super(ProposalTargetQuadrangleOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._cfg = cfg
        # FG_FRACTION = 0.25
        self._fg_fraction = fg_fraction
        self._output_horizon_rois = output_horizon_rois

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois == -1 or self._batch_rois % self._batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights = \
            sample_rois_rotate(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_boxes)

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        for ind, val in enumerate([rois, labels, bbox_targets, bbox_weights]):
            self.assign(out_data[ind], req[ind], val)

        if self._output_horizon_rois:
            # fyk: get bbox of quadrangle rois: [batch_idx, x1, y1, x2, y2, x3, y3, x4, y4]
            bbox_proposals = get_horizon_minAreaRectangle(rois[:, 1:])
            batch_inds = np.zeros((bbox_proposals.shape[0], 1), dtype=np.float32)
            bbox_proposals = np.hstack((batch_inds, bbox_proposals))
            out_idx = 4
            self.assign(out_data[out_idx], req[out_idx], bbox_proposals.astype(np.float32, copy=False))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('proposal_target_rotate')
class ProposalTargetQuadrangleProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction='0.25', output_horizon_rois='False'):
        super(ProposalTargetQuadrangleProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)
        self._output_horizon_rois = strtobool(output_horizon_rois)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        out_blob = ['rois_output', 'label', 'bbox_target', 'bbox_weight']
        if self._output_horizon_rois:
            out_blob.append('horizon_rois')
        return out_blob

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois

        box_dim = 8
        output_rois_shape = (rois, box_dim + 1)
        label_shape = (rois, )
        bbox_target_shape = (rois, self._num_classes * box_dim)
        bbox_weight_shape = (rois, self._num_classes * box_dim)

        out_shape = [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]
        if self._output_horizon_rois:
            output_horizon_shape = (rois, 4 + 1)
            out_shape.append(output_horizon_shape)

        return [rpn_rois_shape, gt_boxes_shape], \
               out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetQuadrangleOperator(self._num_classes, self._batch_images, self._batch_rois, self._cfg, self._fg_fraction,
                                                self._output_horizon_rois)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
