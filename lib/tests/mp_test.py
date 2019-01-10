import mxnet as mx
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus]
print ctx

def get_symbol():
    data = mx.sym.var("data")
    label = mx.sym.var("softmax_label")
    proj = mx.sym.FullyConnected(data, num_hidden=1, name='fc')
    return mx.sym.Group([proj, label])

# sym = get_symbol()
# feat_sym = sym.get_internals()['fc_output']

import multiprocessing

def f(x):
    return x * x

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
xs = range(10)


if __name__ == '__main__':
    print pool.map(f, xs)
    # for y in pool.imap_unordered(f, xs):
    #     print y,  # may be in any order
