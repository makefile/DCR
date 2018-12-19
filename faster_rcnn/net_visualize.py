
import mxnet as mx
import os
# local import
from symbols import *
from config.config import config, update_config

if __name__ == '__main__':
    cfg_file = os.path.join(os.path.expanduser('~'), 'DCR', 'experiments/faster_rcnn/cfgs/DOTA_rrpn_v1.yaml')
    update_config(cfg_file)
    # sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym_instance = resnet_v1_101_rrpn_v1.resnet_v1_101_rrpn_v1()
    sym = sym_instance.get_symbol(config, is_train=True)

    # Visualize your network
    # mx.viz.plot_network(sym)
    # save as PDF: plot.*
    mx.viz.plot_network(sym).view()
