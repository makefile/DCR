# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by fyk
# --------------------------------------------------------

from dataset import *


def load_quadrangle_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_quadrangle_images(roidb)
    return roidb

