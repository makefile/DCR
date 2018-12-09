# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import pprint
import os

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.loader_quadrangle import QuadrangleTestLoader
from core.tester_quadrangle import Predictor, pred_eval_quadrangle_multiscale, pred_eval_dota, pred_eval_dota_quadrangle
from utils.load_model import load_param
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint

def test_rcnn_dota(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval_dota(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

def merge_dets_to_one_file(path_prefix, scales):
    dst_path = os.path.join(path_prefix, 'test_results')
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    for index in range(1, 501):
        dst_file_path = os.path.join(dst_path, 'res_img_{}.txt'.format(index))
        df = open(dst_file_path, 'w')
        content = ''
        for scale in scales:
            src_path = os.path.join(path_prefix, 'test_{}_results'.format(scale))
            src_file_path = os.path.join(src_path, 'res_img_{}.txt'.format(index))
            sf = open(src_file_path, 'r')
            content += sf.read()
            sf.close()
        df.write(content)
        df.close()

def test_rcnn_dota_quadrangle(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None, draw=False, draw_gt=False):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    if cfg.TEST.DO_MULTISCALE_TEST:
        print "multiscale test!"
        multiscales = np.array(cfg.TEST.MULTISCALE)
        original_scales = cfg.SCALES
        for scale in multiscales:
            print "scale: {}".format(scale)
            cfg.SCALES[0] = (int(original_scales[0][0] * scale), int(original_scales[0][1] * scale))
            # get test data iter
            test_data = QuadrangleTestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

            # load model
            arg_params, aux_params = load_param(prefix, epoch, process=True)

            # infer shape
            data_shape_dict = dict(test_data.provide_data_single)
            sym_instance.infer_shape(data_shape_dict)

            sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

            # decide maximum shape
            data_names = [k[0] for k in test_data.provide_data_single]
            label_names = None
            max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
            if not has_rpn:
                max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

            # create predictor
            predictor = Predictor(sym, data_names, label_names,
                                  context=ctx, max_data_shapes=max_data_shape,
                                  provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                                  arg_params=arg_params, aux_params=aux_params)

            # start detection
            pred_eval_quadrangle_multiscale(scale, predictor, test_data, imdb, cfg, vis=vis, draw=draw, ignore_cache=ignore_cache,
                                 thresh=thresh, logger=logger)
        # merge all different test scale results to one file
        merge_dets_to_one_file(imdb.result_path, multiscales)
        # do polygon nms then in evaluation script

    else:
        # get test data iter
        test_data = QuadrangleTestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

        # load model
        arg_params, aux_params = load_param(prefix, epoch, process=True)

        # infer shape
        data_shape_dict = dict(test_data.provide_data_single)
        sym_instance.infer_shape(data_shape_dict)

        sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

        # decide maximum shape
        data_names = [k[0] for k in test_data.provide_data_single]
        label_names = None
        max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
        if not has_rpn:
            max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

        # create predictor
        predictor = Predictor(sym, data_names, label_names,
                              context=ctx, max_data_shapes=max_data_shape,
                              provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                              arg_params=arg_params, aux_params=aux_params)

        # start detection
        pred_eval_dota_quadrangle(predictor, test_data, imdb, cfg, vis=vis, draw=draw, draw_gt=draw_gt, ignore_cache=ignore_cache,
                             thresh=thresh, logger=logger)
