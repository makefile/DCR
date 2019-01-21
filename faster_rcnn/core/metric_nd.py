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

import mxnet as mx
# import numpy as np
# todo not work yet

def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.ENABLE_OHEM or cfg.TRAIN.END2END:
        pred.append('rcnn_label')
    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.astype('int32')

        # filter with keep_inds
        # zs = mx.nd.zeros(label.shape)
        # os = mx.nd.ones(label.shape)
        # eq_mask = mx.ndarray.where(label != -1 and pred_label == label, os, zs)
        # keep_mask = mx.ndarray.where(label != -1, os, zs)
        # todo -1 should be placed at same context(GPU) as label
        self.sum_metric += mx.nd.sum((label != -1) & (pred_label == label)).asscalar()
        self.num_inst += mx.nd.sum(label != -1).asscalar()
        # asscalar is same as self.asnumpy()[0]


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.reshape(-1,).astype('int32')

        # filter with keep_inds
        self.sum_metric += mx.nd.sum((label != -1) & (pred_label == label)).asscalar()
        self.num_inst += mx.nd.sum(label != -1).asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')]

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')]
        num_inst = mx.ndarray.sum(label != -1)

        self.sum_metric += mx.ndarray.sum(bbox_loss).asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')]
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')]
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')]
            else:
                label = labels[self.label.index('rcnn_label')]

        # calculate num_inst (average on those kept anchors)
        num_inst = mx.nd.sum(label != -1)

        self.sum_metric += mx.nd.sum(bbox_loss).asscalar()
        self.num_inst += num_inst.asscalar()
