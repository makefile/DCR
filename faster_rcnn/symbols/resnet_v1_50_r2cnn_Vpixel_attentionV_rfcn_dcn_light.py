# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *


class resnet_v1_50_r2cnn_Vpixel_attentionV_rfcn_dcn_light(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        # self.units = (3, 4, 23, 3) # use for 101
        self.units = (3, 4, 6, 3)  # use for 50
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_v1_conv4(self, data):
        # fyk: ResNet-50-v1 differ from 101 that conv1 has bias
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=False)
                                      # no_bias=True) # for ResNet-101-v1
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
        res3b_branch2a = mx.symbol.Convolution(name='res3b_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2a = mx.symbol.BatchNorm(name='bn3b_branch2a', data=res3b_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2a = bn3b_branch2a
        res3b_branch2a_relu = mx.symbol.Activation(name='res3b_branch2a_relu', data=scale3b_branch2a, act_type='relu')
        res3b_branch2b = mx.symbol.Convolution(name='res3b_branch2b', data=res3b_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b_branch2b = mx.symbol.BatchNorm(name='bn3b_branch2b', data=res3b_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2b = bn3b_branch2b
        res3b_branch2b_relu = mx.symbol.Activation(name='res3b_branch2b_relu', data=scale3b_branch2b, act_type='relu')
        res3b_branch2c = mx.symbol.Convolution(name='res3b_branch2c', data=res3b_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2c = mx.symbol.BatchNorm(name='bn3b_branch2c', data=res3b_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2c = bn3b_branch2c
        res3b = mx.symbol.broadcast_add(name='res3b', *[res3a_relu, scale3b_branch2c])
        res3b_relu = mx.symbol.Activation(name='res3b_relu', data=res3b, act_type='relu')
        res3c_branch2a = mx.symbol.Convolution(name='res3c_branch2a', data=res3b_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2a = mx.symbol.BatchNorm(name='bn3c_branch2a', data=res3c_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2a = bn3c_branch2a
        res3c_branch2a_relu = mx.symbol.Activation(name='res3c_branch2a_relu', data=scale3c_branch2a, act_type='relu')
        res3c_branch2b = mx.symbol.Convolution(name='res3c_branch2b', data=res3c_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3c_branch2b = mx.symbol.BatchNorm(name='bn3c_branch2b', data=res3c_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2b = bn3c_branch2b
        res3c_branch2b_relu = mx.symbol.Activation(name='res3c_branch2b_relu', data=scale3c_branch2b, act_type='relu')
        res3c_branch2c = mx.symbol.Convolution(name='res3c_branch2c', data=res3c_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2c = mx.symbol.BatchNorm(name='bn3c_branch2c', data=res3c_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2c = bn3c_branch2c
        res3c = mx.symbol.broadcast_add(name='res3c', *[res3b_relu, scale3c_branch2c])
        res3c_relu = mx.symbol.Activation(name='res3c_relu', data=res3c, act_type='relu')
        res3d_branch2a = mx.symbol.Convolution(name='res3d_branch2a', data=res3c_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2a = mx.symbol.BatchNorm(name='bn3d_branch2a', data=res3d_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2a = bn3d_branch2a
        res3d_branch2a_relu = mx.symbol.Activation(name='res3d_branch2a_relu', data=scale3d_branch2a, act_type='relu')
        res3d_branch2b = mx.symbol.Convolution(name='res3d_branch2b', data=res3d_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3d_branch2b = mx.symbol.BatchNorm(name='bn3d_branch2b', data=res3d_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2b = bn3d_branch2b
        res3d_branch2b_relu = mx.symbol.Activation(name='res3d_branch2b_relu', data=scale3d_branch2b, act_type='relu')
        res3d_branch2c = mx.symbol.Convolution(name='res3d_branch2c', data=res3d_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2c = mx.symbol.BatchNorm(name='bn3d_branch2c', data=res3d_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2c = bn3d_branch2c
        res3d = mx.symbol.broadcast_add(name='res3d', *[res3c_relu, scale3d_branch2c])
        res3d_relu = mx.symbol.Activation(name='res3d_relu', data=res3d, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3d_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3d_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b_branch2a = mx.symbol.Convolution(name='res4b_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2a = mx.symbol.BatchNorm(name='bn4b_branch2a', data=res4b_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2a = bn4b_branch2a
        res4b_branch2a_relu = mx.symbol.Activation(name='res4b_branch2a_relu', data=scale4b_branch2a, act_type='relu')
        res4b_branch2b = mx.symbol.Convolution(name='res4b_branch2b', data=res4b_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b_branch2b = mx.symbol.BatchNorm(name='bn4b_branch2b', data=res4b_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2b = bn4b_branch2b
        res4b_branch2b_relu = mx.symbol.Activation(name='res4b_branch2b_relu', data=scale4b_branch2b, act_type='relu')
        res4b_branch2c = mx.symbol.Convolution(name='res4b_branch2c', data=res4b_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2c = mx.symbol.BatchNorm(name='bn4b_branch2c', data=res4b_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2c = bn4b_branch2c
        res4b = mx.symbol.broadcast_add(name='res4b', *[res4a_relu, scale4b_branch2c])
        res4b_relu = mx.symbol.Activation(name='res4b_relu', data=res4b, act_type='relu')
        res4c_branch2a = mx.symbol.Convolution(name='res4c_branch2a', data=res4b_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2a = mx.symbol.BatchNorm(name='bn4c_branch2a', data=res4c_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2a = bn4c_branch2a
        res4c_branch2a_relu = mx.symbol.Activation(name='res4c_branch2a_relu', data=scale4c_branch2a, act_type='relu')
        res4c_branch2b = mx.symbol.Convolution(name='res4c_branch2b', data=res4c_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4c_branch2b = mx.symbol.BatchNorm(name='bn4c_branch2b', data=res4c_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2b = bn4c_branch2b
        res4c_branch2b_relu = mx.symbol.Activation(name='res4c_branch2b_relu', data=scale4c_branch2b, act_type='relu')
        res4c_branch2c = mx.symbol.Convolution(name='res4c_branch2c', data=res4c_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2c = mx.symbol.BatchNorm(name='bn4c_branch2c', data=res4c_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2c = bn4c_branch2c
        res4c = mx.symbol.broadcast_add(name='res4c', *[res4b_relu, scale4c_branch2c])
        res4c_relu = mx.symbol.Activation(name='res4c_relu', data=res4c, act_type='relu')
        res4d_branch2a = mx.symbol.Convolution(name='res4d_branch2a', data=res4c_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2a = mx.symbol.BatchNorm(name='bn4d_branch2a', data=res4d_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2a = bn4d_branch2a
        res4d_branch2a_relu = mx.symbol.Activation(name='res4d_branch2a_relu', data=scale4d_branch2a, act_type='relu')
        res4d_branch2b = mx.symbol.Convolution(name='res4d_branch2b', data=res4d_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4d_branch2b = mx.symbol.BatchNorm(name='bn4d_branch2b', data=res4d_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2b = bn4d_branch2b
        res4d_branch2b_relu = mx.symbol.Activation(name='res4d_branch2b_relu', data=scale4d_branch2b, act_type='relu')
        res4d_branch2c = mx.symbol.Convolution(name='res4d_branch2c', data=res4d_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2c = mx.symbol.BatchNorm(name='bn4d_branch2c', data=res4d_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2c = bn4d_branch2c
        res4d = mx.symbol.broadcast_add(name='res4d', *[res4c_relu, scale4d_branch2c])
        res4d_relu = mx.symbol.Activation(name='res4d_relu', data=res4d, act_type='relu')
        res4e_branch2a = mx.symbol.Convolution(name='res4e_branch2a', data=res4d_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2a = mx.symbol.BatchNorm(name='bn4e_branch2a', data=res4e_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2a = bn4e_branch2a
        res4e_branch2a_relu = mx.symbol.Activation(name='res4e_branch2a_relu', data=scale4e_branch2a, act_type='relu')
        res4e_branch2b = mx.symbol.Convolution(name='res4e_branch2b', data=res4e_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4e_branch2b = mx.symbol.BatchNorm(name='bn4e_branch2b', data=res4e_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2b = bn4e_branch2b
        res4e_branch2b_relu = mx.symbol.Activation(name='res4e_branch2b_relu', data=scale4e_branch2b, act_type='relu')
        res4e_branch2c = mx.symbol.Convolution(name='res4e_branch2c', data=res4e_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2c = mx.symbol.BatchNorm(name='bn4e_branch2c', data=res4e_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2c = bn4e_branch2c
        res4e = mx.symbol.broadcast_add(name='res4e', *[res4d_relu, scale4e_branch2c])
        res4e_relu = mx.symbol.Activation(name='res4e_relu', data=res4e, act_type='relu')
        res4f_branch2a = mx.symbol.Convolution(name='res4f_branch2a', data=res4e_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2a = mx.symbol.BatchNorm(name='bn4f_branch2a', data=res4f_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2a = bn4f_branch2a
        res4f_branch2a_relu = mx.symbol.Activation(name='res4f_branch2a_relu', data=scale4f_branch2a, act_type='relu')
        res4f_branch2b = mx.symbol.Convolution(name='res4f_branch2b', data=res4f_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4f_branch2b = mx.symbol.BatchNorm(name='bn4f_branch2b', data=res4f_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2b = bn4f_branch2b
        res4f_branch2b_relu = mx.symbol.Activation(name='res4f_branch2b_relu', data=scale4f_branch2b, act_type='relu')
        res4f_branch2c = mx.symbol.Convolution(name='res4f_branch2c', data=res4f_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2c = mx.symbol.BatchNorm(name='bn4f_branch2c', data=res4f_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2c = bn4f_branch2c
        res4f = mx.symbol.broadcast_add(name='res4f', *[res4e_relu, scale4f_branch2c])
        res4f_relu = mx.symbol.Activation(name='res4f_relu', data=res4f, act_type='relu')
        return res4f_relu

    def get_resnet_v1_conv5(self, conv_feat):
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data = res5b_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu, offset=res5b_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def InceptionModule(self, conv_feat, name):
        '''
        construct Inception module
        :param conv_feat:
        :param name: specify prefix name of the symbols
        :return:
        '''
        # branch1
        tower_13 = mx.symbol.Convolution(data=conv_feat, num_filter=192, name=name + '_mixed_conv_1_1x1',
                                          pad=(0, 0), kernel=(1, 1), stride=(1, 1))
        tower_13 = mx.sym.Activation(data=tower_13, act_type='relu', name=name + '_relu_mixed_conv_1_1x1')
        tower_13 = mx.symbol.Convolution(data=tower_13, num_filter=224, name=name + '_mixed_conv_1_1x3',
                                          pad=(0, 1), kernel=(1, 3), stride=(1, 1))
        tower_13 = mx.symbol.Convolution(data=tower_13, num_filter=256, name=name + '_mixed_conv_1_3x1',
                                          pad=(1, 0), kernel=(3, 1), stride=(1, 1))
        # branch2
        tower_157 = mx.symbol.Convolution(data=conv_feat, num_filter=192, name=name + '_mixed_conv_2_1x1',
                                          pad=(0, 0), kernel=(1, 1), stride=(1, 1))
        tower_157 = mx.sym.Activation(data=tower_157, act_type='relu', name=name + '_relu_mixed_conv_2_1x1')
        tower_157 = mx.symbol.Convolution(data=tower_157, num_filter=192, name=name + '_mixed_conv_2_5x1',
                                          pad=(2, 0), kernel=(5, 1), stride=(1, 1))
        tower_157 = mx.symbol.Convolution(data=tower_157, num_filter=224, name=name + '_mixed_conv_2_1x5',
                                          pad=(0, 2), kernel=(1, 5), stride=(1, 1))
        tower_157 = mx.sym.Activation(data=tower_157, act_type='relu', name=name + '_relu_mixed_conv_2_5x5')
        tower_157 = mx.symbol.Convolution(data=tower_157, num_filter=224, name=name + '_mixed_conv_2_7x1',
                                          pad=(3, 0), kernel=(7, 1), stride=(1, 1))
        tower_157 = mx.symbol.Convolution(data=tower_157, num_filter=256, name=name + '_mixed_conv_2_1x7',
                                          pad=(0, 3), kernel=(1, 7), stride=(1, 1))
        # branch 3
        pooling = mx.sym.Pooling(data=conv_feat, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                 pool_type="avg", name=name + '_pool_avg_3x3')
        pool_conv_proj = mx.symbol.Convolution(data=pooling, num_filter=128, name=name + '_pool_conv_1x1',
                                pad=(0, 0), kernel=(1, 1), stride=(1, 1))
        # branch 4
        conv_proj = mx.symbol.Convolution(data=conv_feat, num_filter=384, name=name + '_cproj',
                                          pad=(0, 0), kernel=(1, 1), stride=(1, 1))

        # concat dim 1024 same as ResNet conv4
        concat = mx.sym.Concat(*[tower_13, tower_157, pool_conv_proj, conv_proj], name='%s_concat' % name)
        channels = 1024
        return concat, channels

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

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
        conv_feat = self.get_resnet_v1_conv4(data)
        #  the multi-dimensional attention network (MDA-Net), feature shape do not change
        # ------ pixel attention: Inception
        f3_incept, _ = self.InceptionModule(conv_feat, name='incept2')
        # saliency map, shape [N, 2, h, w]
        sal_map = mx.symbol.Convolution(name='conv_f3_sal', data=f3_incept, num_filter=2,
                                pad=(0, 0), kernel=(1, 1), stride=(1, 1))
        #  softmax(FG vs BG) only keep FG channel
        sal_prob = mx.sym.softmax(data=sal_map, axis=1, name='sal_prob_softmax')
        # the first set are background probabilities, shape [N, 1, h, w]
        bin_mask_pred = mx.sym.slice_axis(sal_prob, axis=1, begin=1, end=2)
        # apply pixel attention
        conv_feat = mx.symbol.broadcast_mul(conv_feat, bin_mask_pred)

        # res5
        relu1 = self.get_resnet_v1_conv5(conv_feat)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        if is_train:
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 9), name='gt_boxes_reshape')
            # todo the pixel attention loss
            # F.binary_cross_entropy(bin_mask_pred, bin_mask_gt)
            bin_mask_gt = mx.sym.Custom(
                bin_mask_pred=bin_mask_pred, gt_boxes=gt_boxes_reshape, name='bin_mask_gt',
                op_type='BinaryMaskGt', spatial_scale=1. / 16)
            ce_loss = -(mx.sym.log(bin_mask_pred + self.eps) * bin_mask_gt +
                        mx.sym.log(1. - bin_mask_pred + self.eps) * (1. - bin_mask_gt))
            binary_mask_loss = mx.sym.MakeLoss(name='binary_mask_loss', data=ce_loss,
                                               normalization = 'valid', # 'null', 'batch',
                                               grad_scale=cfg.TRAIN.BINARY_MASK_LOSS_WEIGHT)

            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            # gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 9), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight, bbox_target_h, bbox_weight_h = mx.sym.Custom(rois=rois,
                                                gt_boxes=gt_boxes_reshape,
                                                op_type='proposal_target_quadrangle',
                                                num_classes=num_reg_classes,
                                                batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                cfg=cPickle.dumps(cfg),
                                                fg_fraction=cfg.TRAIN.FG_FRACTION,
                                                output_horizon_target=True)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # conv_new_1
        # conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=3.0)
        # relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

        # start light head
        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(15, 1), pad=(7, 0), num_filter=256,
                                        name="conv_new_1", lr_mult=3.0)
        # relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')
        conv_new_2 = mx.sym.Convolution(data=conv_new_1, kernel=(1, 15), pad=(0, 7), num_filter=10 * 7 * 7,
                                        name="conv_new_2", lr_mult=3.0)
        # relu_new_2 = mx.sym.Activation(data=conv_new_2, act_type='relu', name='relu2')
        conv_new_3 = mx.sym.Convolution(data=relu1, kernel=(1, 15), pad=(0, 7), num_filter=256,
                                        name="conv_new_3", lr_mult=3.0)
        # relu_new_3 = mx.sym.Activation(data=conv_new_3, act_type='relu', name='relu3')
        conv_new_4 = mx.sym.Convolution(data=conv_new_3, kernel=(15, 1), pad=(7, 0), num_filter=10 * 7 * 7,
                                        name="conv_new_4", lr_mult=3.0)
        # relu_new_4 = mx.sym.Activation(data=conv_new_4, act_type='relu', name='relu4')
        light_head = mx.symbol.broadcast_add(name='light_head', *[conv_new_2, conv_new_4])
        relu_new_1 = mx.sym.Activation(data=light_head, act_type='relu', name='light_head_relu')
        # end light head

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*8*num_reg_classes, name="rfcn_bbox")
        # trans_cls / trans_cls
        rfcn_cls_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
        rfcn_bbox_offset_t = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")

        rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
        rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                 sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)

        psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
                                                                     output_dim=num_classes, spatial_scale=0.0625, part_size=7)
        psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1,
                                                                     output_dim=8*2, spatial_scale=0.0625, part_size=7)
        # cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        # bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        # RCM pooling
        cls_score_h = mx.sym.Pooling(name='ave_cls_scors_rois_h_', data=psroipooled_cls_rois, pool_type='max', kernel=(7, 1), stride=(7,1))
        cls_score_h = mx.sym.Pooling(name='ave_cls_scors_rois_h', data=cls_score_h, pool_type='avg', kernel=(1, 7), stride=(1, 7))
        cls_score_w = mx.sym.Pooling(name='ave_cls_scors_rois_w_', data=psroipooled_cls_rois, pool_type='max', kernel=(1, 7), stride=(1, 7))
        cls_score_w = mx.sym.Pooling(name='ave_cls_scors_rois_w', data=cls_score_w, pool_type='avg', kernel=(7, 1), stride=(7, 1))
        cls_score = mx.symbol.broadcast_add(name='ave_cls_scors_rois', *[cls_score_h, cls_score_w])
        bbox_pred_h = mx.sym.Pooling(name='ave_bbox_pred_rois_h_', data=psroipooled_loc_rois, pool_type='max', kernel=(7, 1), stride=(7, 1))
        bbox_pred_h = mx.sym.Pooling(name='ave_bbox_pred_rois_h', data=bbox_pred_h, pool_type='avg', kernel=(1, 7), stride=(1, 7))
        bbox_pred_w = mx.sym.Pooling(name='ave_bbox_pred_rois_w_', data=psroipooled_loc_rois, pool_type='max', kernel=(1, 7), stride=(1, 7))
        bbox_pred_w = mx.sym.Pooling(name='ave_bbox_pred_rois_w', data=bbox_pred_w, pool_type='avg', kernel=(7, 1), stride=(7, 1))
        bbox_pred = mx.symbol.broadcast_add(name='ave_bbox_pred_rois', *[bbox_pred_h, bbox_pred_w])

        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 8 * num_reg_classes))

        """
        # light head
        roi_pool = mx.contrib.sym.PSROIPooling(name='roi_pool', data=light_head, rois=rois, group_size=7, pooled_size=7,
                                               output_dim=10, spatial_scale=0.0625)
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=2048)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_1_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_1_relu, num_hidden=num_reg_classes * 4)
        """
        output_horizon = True
        if is_train or output_horizon:
            # horizon_branch, the output_dim can be any num
            roi_pool = mx.contrib.sym.PSROIPooling(name='roi_pool', data=light_head, rois=rois, group_size=7,
                                                   pooled_size=7, output_dim=10, spatial_scale=0.0625)
            fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=2048)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
            cls_score_h = mx.symbol.FullyConnected(name='cls_score_h', data=fc_new_1_relu, num_hidden=num_classes)
            bbox_pred_h = mx.symbol.FullyConnected(name='bbox_pred_h', data=fc_new_1_relu,
                                                   num_hidden=num_reg_classes * 4)

        if is_train:
            # quadrangle rotate branch
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 8 * num_reg_classes), name='bbox_loss_reshape')

            # horizon branch
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem_h, bbox_weights_ohem_h = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                                   num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                                   cls_score=cls_score_h, bbox_pred=bbox_pred_h, labels=label,
                                                                   bbox_targets=bbox_target_h, bbox_weights=bbox_weight_h)
                cls_prob_h = mx.sym.SoftmaxOutput(name='cls_prob_h', data=cls_score_h, label=labels_ohem_h, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_h_ = bbox_weights_ohem_h * mx.sym.smooth_l1( name='bbox_loss_h_', scalar=1.0, data=(bbox_pred_h - bbox_target_h))
                bbox_loss_h = mx.sym.MakeLoss(name='bbox_loss_h', data=bbox_loss_h_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label_h = labels_ohem_h
            else:
                cls_prob_h = mx.sym.SoftmaxOutput(name='cls_prob_h', data=cls_score_h, label=label, normalization='valid')
                bbox_loss_h_ = bbox_weight_h * mx.sym.smooth_l1(name='bbox_loss_h_', scalar=1.0, data=(bbox_pred_h - bbox_target_h))
                bbox_loss_h = mx.sym.MakeLoss(name='bbox_loss_h', data=bbox_loss_h_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label_h = label

            # reshape output
            rcnn_label_h = mx.sym.Reshape(data=rcnn_label_h, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_h_reshape')
            cls_prob_h = mx.sym.Reshape(data=cls_prob_h, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_h_reshape')
            bbox_loss_h = mx.sym.Reshape(data=bbox_loss_h, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_h_reshape')

            # group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label),
                                  binary_mask_loss, mx.sym.BlockGrad(bin_mask_gt),
                                  cls_prob_h, bbox_loss_h, mx.sym.BlockGrad(rcnn_label_h)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 8 * num_reg_classes),
                                       name='bbox_pred_reshape')

            sym_list = [rois, cls_prob, bbox_pred]
            if output_horizon:
                cls_prob_h = mx.sym.SoftmaxActivation(name='cls_prob_h', data=cls_score_h)
                cls_prob_h = mx.sym.Reshape(data=cls_prob_h, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                          name='cls_prob_reshape_h')
                bbox_pred_h = mx.sym.Reshape(data=bbox_pred_h, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                           name='bbox_pred_reshape_h')
                sym_list.extend([cls_prob_h, bbox_pred_h])

            group = mx.sym.Group(sym_list)

        self.sym = group
        return group

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight_rfcn(self, cfg, arg_params, aux_params):
        arg_params['res5a_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_weight'])
        arg_params['res5a_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_bias'])
        arg_params['res5b_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_weight'])
        arg_params['res5b_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_bias'])
        arg_params['res5c_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_weight'])
        arg_params['res5c_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_bias'])

        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])
        arg_params['rfcn_cls_offset_t_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_offset_t_weight'])
        arg_params['rfcn_cls_offset_t_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_offset_t_bias'])
        arg_params['rfcn_bbox_offset_t_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_offset_t_weight'])
        arg_params['rfcn_bbox_offset_t_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_offset_t_bias'])

        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['conv_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_2_weight'])
        arg_params['conv_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_2_bias'])
        arg_params['conv_new_3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_3_weight'])
        arg_params['conv_new_3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_3_bias'])
        arg_params['conv_new_4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_4_weight'])
        arg_params['conv_new_4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_4_bias'])
        # arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        # arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        # arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        # arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        # arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        # arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

        # horizon branch
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['cls_score_h_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_h_weight'])
        arg_params['cls_score_h_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_h_bias'])
        arg_params['bbox_pred_h_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_h_weight'])
        arg_params['bbox_pred_h_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_h_bias'])

    def init_weight_attention_net(self, cfg, arg_params, aux_params):
        # Inception Module
        names = ['_mixed_conv_1_1x1', '_mixed_conv_1_1x3', '_mixed_conv_1_3x1',
                 '_mixed_conv_2_1x1', '_mixed_conv_2_5x1', '_mixed_conv_2_1x5', '_mixed_conv_2_7x1',
                 '_mixed_conv_2_1x7', '_pool_conv_1x1', '_cproj']
        for nm in names:
            prefix = 'incept2' + nm
            arg_params[prefix + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix + '_weight'])
            arg_params[prefix + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix + '_bias'])

        arg_params['conv_f3_sal_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_f3_sal_weight'])
        arg_params['conv_f3_sal_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_f3_sal_bias'])
        # arg_params['se_excitation1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['se_excitation1_weight'])
        # arg_params['se_excitation1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['se_excitation1_bias'])
        # arg_params['se_excitation2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['se_excitation2_weight'])
        # arg_params['se_excitation2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['se_excitation2_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rfcn(cfg, arg_params, aux_params)
        self.init_weight_attention_net(cfg, arg_params, aux_params)
