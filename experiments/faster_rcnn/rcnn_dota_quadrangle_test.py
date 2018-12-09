# --------------------------------------------------------
# Deformable Convolutional Networks
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
#os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))

import test_dota_quadrangle

if __name__ == "__main__":

    test_dota_quadrangle.main()
