# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1' # disable line buffing
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0' # disable select conv algorithm on cudnn
os.environ['MXNET_ENABLE_GPU_P2P'] = '0' # disable multi-node
#os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine' # for debug
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'rfcn_dcr'))

import train_end2end
import test

if __name__ == "__main__":
    train_end2end.main()
    test.main()




