# DCR

## Introduction

**Decoupled Classification Refinement** is initially described in an [ECCV 2018 paper](https://arxiv.org/abs/1803.06799) (DCR V1).
It is further extended (we call it DCR V2) in a recent [tech report](https://arxiv.org/abs/1810.04002). 

## Disclaimer

  It is worth noticing that:

  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository is based on [Deformable ConvNets](https://github.com/msracver/Deformable-ConvNets) and [Decoupled Classification Refinement](https://arxiv.org/abs/1810.04002).

## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on MXNet version 1.3.0 installed using pip. Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release.

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not support Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.

3. Python packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install -r requirements.txt
	```
4. For Windows users, Visual Studio 2015 is needed to compile cython module.


## Installation

1. Clone the Decoupled Classification Refinement repository, and we'll call the directory that you cloned as ${DCR_ROOT}.
```
git clone https://github.com/bowenc0221/Decoupled-Classification-Refinement.git
```

2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet following [this link](http://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU)


## Preparation for Training & Testing

1. Please download **COCO2017** trainval datasets (**Note:** although COCO2014 and COCO2017 has exactly same images, their naming for images are different), and make sure it looks like this:

	```
	./data/coco/
	```

2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```

## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/faster_rcnn_dcr/cfgs`, `./experiments/fpn_dcr/cfgs`.
2. Eight config files have been provided so far, namely, Faster R-CNN(2fc) for COCO, Deformable Faster R-CNN(2fc) for COCO, FPN for COCO, Deformable FPN for COCO, respectively and their DCR versions. We use 4 GPUs to train all models on COCO.

3. To perform experiments, run the python scripts with the corresponding config file as input. For example, to train and test deformable convnets + DCR on COCO with ResNet-v1-101, use the following command
    ```
    python experiments/faster_rcnn_dcr/rcnn_end2end_train_test.py --cfg experiments/faster_rcnn_dcr/cfgs/resnet_v1_101_coco_trainval_dcn_dcr_end2end.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `output/dcn_dcr/coco/`. (Note: the command above automatically run test after training)  
    To only test the model, use command
    ```
    python experiments/faster_rcnn_dcr/rcnn_test.py --cfg experiments/faster_rcnn_dcr/cfgs/resnet_v1_101_coco_trainval_dcn_dcr_end2end.yaml
    ```
4. Please find more details in config files and in our code.


## License

© Licensed under an MIT license.

## Citing DCR

If you find Decoupled Classification Refinement module useful in your research, please consider citing:
```
@article{cheng18decoupled,
author = {Cheng, Bowen and Wei, Yunchao and Shi, Honghui and Feris, Rogerio and Xiong, Jinjun and Huang, Thomas},
title = {Decoupled Classification Refinement: Hard False Positive Suppression for Object Detection},
journal = {arXiv preprint arXiv:1810.04002},
year = {2018}
}

@inproceedings{cheng18revisiting,
author = {Cheng, Bowen and Wei, Yunchao and Shi, Honghui and Feris, Rogerio and Xiong, Jinjun and Huang, Thomas},
title = {Revisiting RCNN: On Awakening the Classification Power of Faster RCNN},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

