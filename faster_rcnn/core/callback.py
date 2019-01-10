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

import time
import logging
import mxnet as mx
import sys


class Speedometer(object):
    def __init__(self, batch_size, frequent=50, num_epoch=-1):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        # add by fyk
        self.num_epoch = num_epoch
        self.start_run_tic = time.time()
        self.total_run_batches = 0
        self.total_nbatch = 0

    def __call__(self, param):
        """Callback to Show speed."""
        self.total_run_batches += 1
        count = param.nbatch
        if self.last_count > count:
            self.init = False
            # epoch end
            self.total_nbatch = self.num_epoch * self.last_count
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                now = time.time()
                secs = now - self.tic
                total_run_secs = now - self.start_run_tic
                remain_secs = (self.total_nbatch - self.total_run_batches) * total_run_secs / self.total_run_batches
                speed = self.frequent * self.batch_size / secs
                # s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    if self.num_epoch > 0:
                        s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTime: [%d+%d h]\tTrain-" % (param.epoch, count, speed,
                                                                        int(total_run_secs/3600), int(remain_secs/3600))
                    else:
                        s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)

                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                print(s)
                # fyk: flush screen output
                # s = '\r' + s
                # sys.stdout.write(s)
                # sys.stdout.flush()
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        # fyk modify
        # if 'bbox_pred_h_weight' in arg:
            # horizon branch of R2CNN
            # since our means are 0s, and stds are 1s, so we do not need to unnormalize
        # Faster R-CNN
        if 'bbox_pred_weight' in arg:
            arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
            arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
            arg.pop('bbox_pred_weight_test')
            arg.pop('bbox_pred_bias_test')
        # R-FCN
        elif 'rfcn_bbox_weight' in arg:
            weight = arg['rfcn_bbox_weight']
            bias = arg['rfcn_bbox_bias']
            repeat = bias.shape[0] / means.shape[0]

            arg['rfcn_bbox_weight_test'] = weight * mx.nd.repeat(mx.nd.array(stds), repeats=repeat).reshape(
                (bias.shape[0], 1, 1, 1))
            arg['rfcn_bbox_bias_test'] = arg['rfcn_bbox_bias'] * mx.nd.repeat(mx.nd.array(stds), repeats=repeat) \
                                         + mx.nd.repeat(mx.nd.array(means), repeats=repeat)
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
            arg.pop('rfcn_bbox_weight_test')
            arg.pop('rfcn_bbox_bias_test')
    return _callback
