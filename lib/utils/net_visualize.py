
import mxnet as mx
import os
# local import
from symbols import *
from config.config import config, update_config

def plot_viz_net(cfg_file):
    update_config(cfg_file)
    # sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym_instance = resnet_v1_101_rrpn_v1.resnet_v1_101_rrpn_v1()
    sym = sym_instance.get_symbol(config, is_train=True)

    # Visualize your network
    # mx.viz.plot_network(sym)
    # save as PDF: plot.*
    mx.viz.plot_network(sym).view()

def plot_mxboard_graph(cfg_file, logdir):
    from mxboard import SummaryWriter
    update_config(cfg_file)
    # sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym_instance = resnet_v1_101_rrpn_v1.resnet_v1_101_rrpn_v1()
    sym = sym_instance.get_symbol(config, is_train=True)
    with SummaryWriter(logdir=logdir) as sw:
        sw.add_graph(sym)


def viz_test():
    cfg_file = os.path.join(os.path.expanduser('~'), 'DCR', 'experiments/faster_rcnn/cfgs/DOTA_rrpn_v1.yaml')
    plot_viz_net(cfg_file)

def mxboard_graph_test():
    cfg_file = os.path.join(os.path.expanduser('~'), 'DCR', 'experiments/faster_rcnn/cfgs/DOTA_rrpn_v1.yaml')
    logdir = os.path.join(os.path.expanduser('~'), 'DCR', './output/logs')
    plot_mxboard_graph(cfg_file, logdir)

if __name__ == '__main__':
    mxboard_graph_test()
