import mxnet as mx

def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512, memonger=False,
                  fix_bn=False):
    """Return ResNet Unit symbol for building ResNet V2
        Parameters
        ----------
        data : str
            Input data
        num_filter : int
            Number of output channels
        stride : tuple
            Stride used in convolution
        dim_match : Boolean
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        fix_bn : bool
            fix bn param
    """
    if fix_bn:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                               pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    if fix_bn:
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
    else:
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                               pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    if fix_bn:
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
    else:
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut

def residual_unit_dilate(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512,
                         memonger=False, fix_bn=False):
    if fix_bn:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                               pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    if fix_bn:
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
    else:
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), dilate=(2, 2),
                               stride=stride, pad=(2, 2),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    if fix_bn:
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
    else:
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut

def residual_unit_deform(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512,
                         memonger=False, fix_bn=False):
    if fix_bn:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                               pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    if fix_bn:
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
    else:
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    offset = mx.symbol.Convolution(name=name + '_offset', data=act2,
                                   num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                   dilate=(2, 2), cudnn_off=True)
    conv2 = mx.contrib.symbol.DeformableConvolution(name=name + '_conv2', data=act2,
                                                    offset=offset,
                                                    num_filter=512, pad=(2, 2), kernel=(3, 3),
                                                    num_deformable_group=4,
                                                    stride=(1, 1), dilate=(2, 2), no_bias=True)
    if fix_bn:
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
    else:
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut

def get_resnet_v2_conv4(data, units, filter_list, bn_mom=0.9, fix_bn=True, workspace=512):
    """
    Construct ResNet symbol
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    workspace : int
        Workspace used in convolution operator
    """
    num_stage = 4
    num_unit = len(units)
    assert (num_unit == num_stage)
    # stage 0
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, use_global_stats=fix_bn,
                            name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=fix_bn, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # stage 1-3
    for i in range(num_stage - 1):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), dim_match=False,
                             name='stage%d_unit%d' % (i + 1, 1), bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), dim_match=True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)

    return body

def get_resnet_v2_conv5(body, units, filter_list, bn_mom=0.9, fix_bn=True, workspace=512,
                        dilate=True, deform=False):

    num_stage = len(units)

    i = num_stage - 1
    if deform:
        body = residual_unit_deform(body, filter_list[i + 1], (1, 1), False, name='stage%d_unit%d' % (i + 1, 1),
                                    bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)
    elif dilate:
        body = residual_unit_dilate(body, filter_list[i + 1], (1, 1), False, name='stage%d_unit%d' % (i + 1, 1),
                                    bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)
    else:
        body = residual_unit(body, filter_list[i + 1], (1, 1), False, name='stage%d_unit%d' % (i + 1, 1),
                             bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)
    for j in range(units[i] - 1):
        if deform:
            body = residual_unit_deform(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                        bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)
        elif dilate:
            body = residual_unit_dilate(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                        bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)
        else:
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bn_mom=bn_mom, workspace=workspace, fix_bn=fix_bn)

    return body



"""
def resnet_18(data, bn_mom=0.99, bn_global=True):
    return resnet(data, units=[2, 2, 2, 2], num_stage=4, filter_list=[64, 64, 128, 256, 512],
                  bn_mom=bn_mom, bn_global=bn_global, workspace=512)

def resnet_34(data, bn_mom=0.99, bn_global=True):
    return resnet(data, units=[3, 4, 6, 3], num_stage=4, filter_list=[64, 64, 128, 256, 512],
                  bn_mom=bn_mom, bn_global=bn_global, workspace=512)

def resnet_50(data, bn_mom=0.99, bn_global=True):
    return resnet(data, units=[3, 4, 6, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048],
                  bn_mom=bn_mom, bn_global=bn_global, workspace=512)

def resnet_101(data, bn_mom=0.99, bn_global=True):
    return resnet(data, units=[3, 4, 23, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048],
                  bn_mom=bn_mom, bn_global=bn_global, workspace=512)

def resnet_152(data, bn_mom=0.99, bn_global=True):
    return resnet(data, units=[3, 8, 36, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048],
                  bn_mom=bn_mom, bn_global=bn_global, workspace=512)

def resnet_200(data, bn_mom=0.99, bn_global=True):
    return resnet(data, units=[3, 24, 36, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048],
                  bn_mom=bn_mom, bn_global=bn_global, workspace=512)
"""
