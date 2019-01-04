import time
import numpy as np
import sys, os
my_mxnet_path = os.path.join(os.path.expanduser('~'),
                             'miniconda2/lib/python2.7/site-packages/mxnet-0.10.1-py2.7.egg/')

sys.path.insert(0, my_mxnet_path)
import mxnet as mx

'''
A MCVE for check whether the CustomOp of Python runs parallelly
MCVE means a Minimal, Complete, and Verifiable example.
code and related issue: https://github.com/apache/incubator-mxnet/issues/8884
'''
class DebugOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(DebugOperator, self).__init__()
        self.pos = kwargs.get("pos", None)

    def forward(self, is_train, req, in_data, out_data, aux):
        print("enter GPU-%d: %.4f" % (in_data[0][0].context.device_id, time.time()))
        # time.sleep(0.1)
        time.sleep(1)
        self.assign(out_data[0], req[0], 0)
        print("exit  GPU-%d: %.4f" % (in_data[0][0].context.device_id, time.time()))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register("Debug")
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(DebugProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [(1, )]

    def create_operator(self, ctx, shapes, dtypes):
        return DebugOperator(**self._kwargs)


def get_symbol():
    data = mx.sym.var("data")
    label = mx.sym.var("softmax_label")
    proj = mx.sym.FullyConnected(data, num_hidden=1)
    debug = mx.sym.Custom(proj, op_type="Debug", name="debug")
    return mx.sym.Group([debug, label])


if __name__ == "__main__":
    print 'mxnet version %s'%mx.__version__
    gpus = [0, 1]
    sym = get_symbol()
    mod = mx.module.Module(sym, context=[mx.gpu(i) for i in gpus])
    mod.bind(data_shapes=[("data", (len(gpus), 1))], label_shapes=[("softmax_label", (len(gpus), 1))])
    dummy_data = np.zeros((60, 1))
    data = mx.io.NDArrayIter(data=dummy_data, label=dummy_data, batch_size=len(gpus))
    tic = time.time()
    print("start: %.4f" % tic)
    mod.fit(data, num_epoch=1, eval_metric=mx.metric.Loss(output_names=["debug_output"]))
    print("cost: %d s" % (time.time() - tic))