"""
get pretrained model from gluon-cv model zoo and save to Module static graph symbol
"""
import mxnet as mx
from mxnet import gluon
# model = gluon.model_zoo.vision.resnet101_v1(pretrained=True)
# model.export('save')
# module = mx.mod.Module.load('save')
net = gluon.model_zoo.vision.resnet50_v1(pretrained=True)
x = mx.nd.random.uniform(shape=(1,3,224,224))
# x = mx.nd.random.uniform(shape=(1,3,224,224),ctx=mx.gpu())
# export net json and param
net.hybridize()
# Please first call block.hybridize() and then run forward with this block at least once before calling export.
net(x)
net.export('resnet_v1_50', epoch=0)
