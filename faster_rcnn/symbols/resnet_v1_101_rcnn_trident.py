# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang, Bin Xiao
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *


class resnet_v1_101_rcnn_trident(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]
        # shared params
        self.shared_pretrained_params = [
            # res4
            'res4a_branch1',
            'res4a_branch2a', 'res4a_branch2b', 'res4a_branch2c',
            'res4b1_branch2a', 'res4b1_branch2b', 'res4b1_branch2c',
            'res4b2_branch2a', 'res4b2_branch2b', 'res4b2_branch2c',
            'res4b3_branch2a', 'res4b3_branch2b', 'res4b3_branch2c',
            'res4b4_branch2a', 'res4b4_branch2b', 'res4b4_branch2c',
            'res4b5_branch2a', 'res4b5_branch2b', 'res4b5_branch2c',
            'res4b6_branch2a', 'res4b6_branch2b', 'res4b6_branch2c',
            'res4b7_branch2a', 'res4b7_branch2b', 'res4b7_branch2c',
            'res4b8_branch2a', 'res4b8_branch2b', 'res4b8_branch2c',
            'res4b9_branch2a', 'res4b9_branch2b', 'res4b9_branch2c',
            'res4b10_branch2a', 'res4b10_branch2b', 'res4b10_branch2c',
            'res4b11_branch2a', 'res4b11_branch2b', 'res4b11_branch2c',
            'res4b12_branch2a', 'res4b12_branch2b', 'res4b12_branch2c',
            'res4b13_branch2a', 'res4b13_branch2b', 'res4b13_branch2c',
            'res4b14_branch2a', 'res4b14_branch2b', 'res4b14_branch2c',
            'res4b15_branch2a', 'res4b15_branch2b', 'res4b15_branch2c',
            'res4b16_branch2a', 'res4b16_branch2b', 'res4b16_branch2c',
            'res4b17_branch2a', 'res4b17_branch2b', 'res4b17_branch2c',
            'res4b18_branch2a', 'res4b18_branch2b', 'res4b18_branch2c',
            'res4b19_branch2a', 'res4b19_branch2b', 'res4b19_branch2c',
            'res4b20_branch2a', 'res4b20_branch2b', 'res4b20_branch2c',
            'res4b21_branch2a', 'res4b21_branch2b', 'res4b21_branch2c',
            'res4b22_branch2a', 'res4b22_branch2b', 'res4b22_branch2c',
            # res5
            'res5a_branch1',
            'res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c',
            'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c',
            'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c'
        ]
        self.shared_param_init_list = [
            # RPN
            'rpn_conv_3x3', 'rpn_cls_score', 'rpn_bbox_pred',
            # RCNN
            'conv_new_1', 'fc_new_1', 'fc_new_2', 'cls_score', 'bbox_pred'
        ]
        self.shared_param_list = self.shared_pretrained_params + self.shared_param_init_list
        self.shared_param_dict = {}
        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight')
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias')

    def get_resnet_v1_conv3(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0), kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0), kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a, act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b, act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a, act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b, act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a, act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b, act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        return res3b3_relu

    def get_resnet_v1_conv4(self, res3b3_relu, dilate=(1,1), prefix=''):
        res4a_branch1 = mx.symbol.Convolution(name=prefix + 'res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True,
                                              weight=self.shared_param_dict['res4a_branch1_weight'])
        bn4a_branch1 = mx.symbol.BatchNorm(name=prefix + 'bn4a_branch1', data=res4a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name=prefix + 'res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True,
                                               weight=self.shared_param_dict['res4a_branch2a_weight'])
        bn4a_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name=prefix + 'res4a_branch2b', data=res4a_branch2a_relu, num_filter=256, pad=dilate,
                                               kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                               weight=self.shared_param_dict['res4a_branch2b_weight'])
        bn4a_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name=prefix + 'res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res4a_branch2c_weight'])
        bn4a_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name=prefix + 'res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name=prefix + 'res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name=prefix + 'res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b1_branch2a_weight'])
        bn4b1_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b1_branch2a_relu', data=scale4b1_branch2a, act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name=prefix + 'res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b1_branch2b_weight'])
        bn4b1_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b1_branch2b_relu', data=scale4b1_branch2b, act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name=prefix + 'res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b1_branch2c_weight'])
        bn4b1_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name=prefix + 'res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name=prefix + 'res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name=prefix + 'res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b2_branch2a_weight'])
        bn4b2_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b2_branch2a_relu', data=scale4b2_branch2a, act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name=prefix + 'res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b2_branch2b_weight'])
        bn4b2_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b2_branch2b_relu', data=scale4b2_branch2b, act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name=prefix + 'res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b2_branch2c_weight'])
        bn4b2_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name=prefix + 'res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name=prefix + 'res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name=prefix + 'res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b3_branch2a_weight'])
        bn4b3_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b3_branch2a_relu', data=scale4b3_branch2a, act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name=prefix + 'res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b3_branch2b_weight'])
        bn4b3_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b3_branch2b_relu', data=scale4b3_branch2b, act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name=prefix + 'res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b3_branch2c_weight'])
        bn4b3_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name=prefix + 'res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name=prefix + 'res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name=prefix + 'res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b4_branch2a_weight'])
        bn4b4_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b4_branch2a_relu', data=scale4b4_branch2a, act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name=prefix + 'res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b4_branch2b_weight'])
        bn4b4_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b4_branch2b_relu', data=scale4b4_branch2b, act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name=prefix + 'res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b4_branch2c_weight'])
        bn4b4_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name=prefix + 'res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name=prefix + 'res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name=prefix + 'res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b5_branch2a_weight'])
        bn4b5_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b5_branch2a_relu', data=scale4b5_branch2a, act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name=prefix + 'res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b5_branch2b_weight'])
        bn4b5_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b5_branch2b_relu', data=scale4b5_branch2b, act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name=prefix + 'res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b5_branch2c_weight'])
        bn4b5_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name=prefix + 'res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name=prefix + 'res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name=prefix + 'res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b6_branch2a_weight'])
        bn4b6_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b6_branch2a_relu', data=scale4b6_branch2a, act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name=prefix + 'res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b6_branch2b_weight'])
        bn4b6_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b6_branch2b_relu', data=scale4b6_branch2b, act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name=prefix + 'res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b6_branch2c_weight'])
        bn4b6_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name=prefix + 'res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name=prefix + 'res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name=prefix + 'res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b7_branch2a_weight'])
        bn4b7_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b7_branch2a_relu', data=scale4b7_branch2a, act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name=prefix + 'res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b7_branch2b_weight'])
        bn4b7_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b7_branch2b_relu', data=scale4b7_branch2b, act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name=prefix + 'res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b7_branch2c_weight'])
        bn4b7_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name=prefix + 'res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name=prefix + 'res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name=prefix + 'res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b8_branch2a_weight'])
        bn4b8_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b8_branch2a_relu', data=scale4b8_branch2a, act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name=prefix + 'res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b8_branch2b_weight'])
        bn4b8_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b8_branch2b_relu', data=scale4b8_branch2b, act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name=prefix + 'res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b8_branch2c_weight'])
        bn4b8_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name=prefix + 'res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name=prefix + 'res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name=prefix + 'res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b9_branch2a_weight'])
        bn4b9_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b9_branch2a_relu', data=scale4b9_branch2a, act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name=prefix + 'res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                weight=self.shared_param_dict['res4b9_branch2b_weight'])
        bn4b9_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b9_branch2b_relu', data=scale4b9_branch2b, act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name=prefix + 'res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                weight=self.shared_param_dict['res4b9_branch2c_weight'])
        bn4b9_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name=prefix + 'res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name=prefix + 'res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name=prefix + 'res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b10_branch2a_weight'])
        bn4b10_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b10_branch2a_relu', data=scale4b10_branch2a, act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name=prefix + 'res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b10_branch2b_weight'])
        bn4b10_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b10_branch2b_relu', data=scale4b10_branch2b, act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name=prefix + 'res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b10_branch2c_weight'])
        bn4b10_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name=prefix + 'res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name=prefix + 'res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name=prefix + 'res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b11_branch2a_weight'])
        bn4b11_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b11_branch2a_relu', data=scale4b11_branch2a, act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name=prefix + 'res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b11_branch2b_weight'])
        bn4b11_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b11_branch2b_relu', data=scale4b11_branch2b, act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name=prefix + 'res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b11_branch2c_weight'])
        bn4b11_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name=prefix + 'res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name=prefix + 'res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name=prefix + 'res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b12_branch2a_weight'])
        bn4b12_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b12_branch2a_relu', data=scale4b12_branch2a, act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name=prefix + 'res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b12_branch2b_weight'])
        bn4b12_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b12_branch2b_relu', data=scale4b12_branch2b, act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name=prefix + 'res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b12_branch2c_weight'])
        bn4b12_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name=prefix + 'res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name=prefix + 'res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name=prefix + 'res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b13_branch2a_weight'])
        bn4b13_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b13_branch2a_relu', data=scale4b13_branch2a, act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name=prefix + 'res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b13_branch2b_weight'])
        bn4b13_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b13_branch2b_relu', data=scale4b13_branch2b, act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name=prefix + 'res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b13_branch2c_weight'])
        bn4b13_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name=prefix + 'res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name=prefix + 'res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name=prefix + 'res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b14_branch2a_weight'])
        bn4b14_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b14_branch2a_relu', data=scale4b14_branch2a, act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name=prefix + 'res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b14_branch2b_weight'])
        bn4b14_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b14_branch2b_relu', data=scale4b14_branch2b, act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name=prefix + 'res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b14_branch2c_weight'])
        bn4b14_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name=prefix + 'res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name=prefix + 'res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name=prefix + 'res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b15_branch2a_weight'])
        bn4b15_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b15_branch2a_relu', data=scale4b15_branch2a, act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name=prefix + 'res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b15_branch2b_weight'])
        bn4b15_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b15_branch2b_relu', data=scale4b15_branch2b, act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name=prefix + 'res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b15_branch2c_weight'])
        bn4b15_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name=prefix + 'res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name=prefix + 'res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name=prefix + 'res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b16_branch2a_weight'])
        bn4b16_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b16_branch2a_relu', data=scale4b16_branch2a, act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name=prefix + 'res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b16_branch2b_weight'])
        bn4b16_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b16_branch2b_relu', data=scale4b16_branch2b, act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name=prefix + 'res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b16_branch2c_weight'])
        bn4b16_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name=prefix + 'res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name=prefix + 'res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name=prefix + 'res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b17_branch2a_weight'])
        bn4b17_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b17_branch2a_relu', data=scale4b17_branch2a, act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name=prefix + 'res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b17_branch2b_weight'])
        bn4b17_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b17_branch2b_relu', data=scale4b17_branch2b, act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name=prefix + 'res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b17_branch2c_weight'])
        bn4b17_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name=prefix + 'res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name=prefix + 'res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name=prefix + 'res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b18_branch2a_weight'])
        bn4b18_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b18_branch2a_relu', data=scale4b18_branch2a, act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name=prefix + 'res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b18_branch2b_weight'])
        bn4b18_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b18_branch2b_relu', data=scale4b18_branch2b, act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name=prefix + 'res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b18_branch2c_weight'])
        bn4b18_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name=prefix + 'res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name=prefix + 'res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name=prefix + 'res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b19_branch2a_weight'])
        bn4b19_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b19_branch2a_relu', data=scale4b19_branch2a, act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name=prefix + 'res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b19_branch2b_weight'])
        bn4b19_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b19_branch2b_relu', data=scale4b19_branch2b, act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name=prefix + 'res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b19_branch2c_weight'])
        bn4b19_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name=prefix + 'res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name=prefix + 'res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name=prefix + 'res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b20_branch2a_weight'])
        bn4b20_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b20_branch2a_relu', data=scale4b20_branch2a, act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name=prefix + 'res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b20_branch2b_weight'])
        bn4b20_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b20_branch2b_relu', data=scale4b20_branch2b, act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name=prefix + 'res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b20_branch2c_weight'])
        bn4b20_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name=prefix + 'res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name=prefix + 'res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name=prefix + 'res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b21_branch2a_weight'])
        bn4b21_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b21_branch2a_relu', data=scale4b21_branch2a, act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name=prefix + 'res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b21_branch2b_weight'])
        bn4b21_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b21_branch2b_relu', data=scale4b21_branch2b, act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name=prefix + 'res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b21_branch2c_weight'])
        bn4b21_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name=prefix + 'res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name=prefix + 'res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name=prefix + 'res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b22_branch2a_weight'])
        bn4b22_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name=prefix + 'res4b22_branch2a_relu', data=scale4b22_branch2a, act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name=prefix + 'res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=dilate, kernel=(3, 3), stride=(1, 1), dilate=dilate, no_bias=True,
                                                 weight=self.shared_param_dict['res4b22_branch2b_weight'])
        bn4b22_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name=prefix + 'res4b22_branch2b_relu', data=scale4b22_branch2b, act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name=prefix + 'res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,
                                                 weight=self.shared_param_dict['res4b22_branch2c_weight'])
        bn4b22_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name=prefix + 'res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name=prefix + 'res4b22_relu', data=res4b22, act_type='relu')
        return res4b22_relu

    def get_resnet_v1_conv5(self, conv_feat, prefix=''):
        res5a_branch1 = mx.symbol.Convolution(name=prefix + 'res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True,
                                              weight=self.shared_param_dict['res5a_branch1_weight'])
        bn5a_branch1 = mx.symbol.BatchNorm(name=prefix + 'bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name=prefix + 'res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res5a_branch2a_weight'])
        bn5a_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name=prefix + 'res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name=prefix + 'res5a_branch2b', data=res5a_branch2a_relu, num_filter=512, pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True,
                                               weight=self.shared_param_dict['res5a_branch2b_weight'])
        bn5a_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name=prefix + 'res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name=prefix + 'res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res5a_branch2c_weight'])
        bn5a_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name=prefix + 'res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name=prefix + 'res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name=prefix + 'res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res5b_branch2a_weight'])
        bn5b_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name=prefix + 'res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name=prefix + 'res5b_branch2b', data=res5b_branch2a_relu, num_filter=512, pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True,
                                               weight=self.shared_param_dict['res5b_branch2b_weight'])
        bn5b_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name=prefix + 'res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name=prefix + 'res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res5b_branch2c_weight'])
        bn5b_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name=prefix + 'res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name=prefix + 'res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name=prefix + 'res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res5c_branch2a_weight'])
        bn5c_branch2a = mx.symbol.BatchNorm(name=prefix + 'bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name=prefix + 'res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name=prefix + 'res5c_branch2b', data=res5c_branch2a_relu, num_filter=512, pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True,
                                               weight=self.shared_param_dict['res5c_branch2b_weight'])
        bn5c_branch2b = mx.symbol.BatchNorm(name=prefix + 'bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name=prefix + 'res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name=prefix + 'res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True,
                                               weight=self.shared_param_dict['res5c_branch2c_weight'])
        bn5c_branch2c = mx.symbol.BatchNorm(name=prefix + 'bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name=prefix + 'res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name=prefix + 'res5c_relu', data=res5c, act_type='relu')
        return res5c_relu

    def get_rpn(self, conv_feat, num_anchors, prefix=''):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name=prefix + "rpn_conv_3x3",
            weight=self.shared_param_dict['rpn_conv_3x3_weight'], bias=self.shared_param_dict['rpn_conv_3x3_bias'])
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name=prefix + "rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name=prefix + "rpn_cls_score",
            weight=self.shared_param_dict['rpn_cls_score_weight'], bias=self.shared_param_dict['rpn_cls_score_bias'])
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name=prefix + "rpn_bbox_pred",
            weight=self.shared_param_dict['rpn_bbox_pred_weight'], bias=self.shared_param_dict['rpn_bbox_pred_bias'])
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True, infer_approx=False):

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv3_feat = self.get_resnet_v1_conv3(data)

        # dialte ratio for different scale
        dilates = (1, 2, 3)
        # be careful about the name, which is used both for RPN data shape infer as well as test phase output indexing
        prefixes = {d: 'd' + str(d) + '_' for d in dilates}
        prefixes[2] = ''

        # parallel branch
        all_blob_list = []
        if is_train:
            for d in dilates:
                blob_list = self.get_branch_symbol(cfg, conv3_feat, im_info, gt_boxes=gt_boxes, rpn_label=rpn_label,
                                                   rpn_bbox_target=rpn_bbox_target, rpn_bbox_weight=rpn_bbox_weight,
                                                   dilate=d, is_train=is_train, prefix=prefixes[d], infer_approx=infer_approx)
                all_blob_list.extend(blob_list)
            group = mx.sym.Group(all_blob_list)
        elif infer_approx: # single branch
            blob_list = self.get_branch_symbol(cfg, conv3_feat, im_info, gt_boxes=None, rpn_label=None,
                                               rpn_bbox_target=None, rpn_bbox_weight=None,
                                               dilate=2, is_train=is_train, prefix=prefixes[2], infer_approx=infer_approx)
            group = mx.sym.Group(blob_list)
        else:
            for d in dilates:
                blob_list = self.get_branch_symbol(cfg, conv3_feat, im_info, gt_boxes=None, rpn_label=None,
                                                   rpn_bbox_target=None, rpn_bbox_weight=None,
                                                   dilate=d, is_train=is_train, prefix=prefixes[d], infer_approx=infer_approx)
                all_blob_list.extend(blob_list)

            # stack each blob at axis 1, [rois, cls_prob, bbox_pred]
            n_blobs = 3
            rois        = mx.sym.Concat(*all_blob_list[0::n_blobs], dim=0, name='rois')
            cls_prob    = mx.sym.Concat(*all_blob_list[1::n_blobs], dim=1, name='cls_prob_reshape')
            bbox_pred   = mx.sym.Concat(*all_blob_list[2::n_blobs], dim=1, name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def get_branch_symbol(self, cfg, conv3_feat, im_info, gt_boxes=None, rpn_label=None,
                          rpn_bbox_target=None, rpn_bbox_weight=None,
                          dilate=1, is_train=True, prefix='', infer_approx=True):
        """
        parallel branch which share same param
        :param dilate: dilate rate for different branch scale
        :return: blob list
        """
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        conv_feat = self.get_resnet_v1_conv4(conv3_feat, dilate=(dilate, dilate), prefix=prefix)
        # res5
        relu1 = self.get_resnet_v1_conv5(conv_feat, prefix=prefix)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors, prefix=prefix)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name=prefix + "rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                   normalization='valid', use_ignore=True, ignore_label=-1, name=prefix + "rpn_cls_prob")

            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name=prefix + 'rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))

            rpn_bbox_loss = mx.sym.MakeLoss(name=prefix + 'rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name=prefix + "rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name=prefix + 'rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name=prefix + 'rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name=prefix + 'rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name=prefix + 'gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name=prefix + "rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name=prefix + "rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name=prefix + 'rpn_cls_prob_reshape')

            if (not infer_approx) and prefix == '': prefix = 'd2_'
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name=prefix + 'rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name=prefix + 'rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

            if (not infer_approx) and prefix == 'd2_': prefix = '' # recover

        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name=prefix + "conv_new_1",
            weight=self.shared_param_dict['conv_new_1_weight'], bias=self.shared_param_dict['conv_new_1_bias'])
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name=prefix + 'conv_new_1_relu')

        roi_pool = mx.symbol.ROIPooling(
            name=prefix + 'roi_pool', data=conv_new_1_relu, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name=prefix + 'fc_new_1', data=roi_pool, num_hidden=1024,
            weight=self.shared_param_dict['fc_new_1_weight'], bias=self.shared_param_dict['fc_new_1_bias'])
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name=prefix + 'fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name=prefix + 'fc_new_2', data=fc_new_1_relu, num_hidden=1024,
            weight=self.shared_param_dict['fc_new_2_weight'], bias=self.shared_param_dict['fc_new_2_bias'])
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name=prefix + 'fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name=prefix + 'cls_score', data=fc_new_2_relu, num_hidden=num_classes,
            weight=self.shared_param_dict['cls_score_weight'], bias=self.shared_param_dict['cls_score_bias'])
        bbox_pred = mx.symbol.FullyConnected(name=prefix + 'bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4,
            weight=self.shared_param_dict['bbox_pred_weight'], bias=self.shared_param_dict['bbox_pred_bias'])

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name=prefix + 'cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name=prefix + 'bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name=prefix + 'bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name=prefix + 'cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name=prefix + 'bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name=prefix + 'bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name=prefix + 'label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name=prefix + 'cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name=prefix + 'bbox_loss_reshape')
            group = [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]
            # group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            if (not infer_approx) and prefix == '': prefix = 'd2_'
            cls_prob = mx.sym.SoftmaxActivation(name=prefix + 'cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name=prefix + 'cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name=prefix + 'bbox_pred_reshape')
            group = [rois, cls_prob, bbox_pred]
            # group = mx.sym.Group([rois, cls_prob, bbox_pred])

        # self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        for name in self.shared_param_init_list:
            arg_params[name + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[name + '_weight'])
            arg_params[name + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_bias'])
        # handle bn params
        for ps in [arg_params, aux_params]:
            for key in ps.keys():
                if 'bn4' in key or 'bn5' in key:
                    for d in ['d1_', 'd3_']:
                        ps[d + key] = ps[key]
        # self.init_weight_rpn(cfg, arg_params, aux_params)
        # self.init_weight_rcnn(cfg, arg_params, aux_params)

