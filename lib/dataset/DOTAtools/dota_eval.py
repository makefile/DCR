#!/usr/bin/env python
import numpy as np
import merge_result
import os, sys
import cv2
from distutils.util import strtobool
from dota_evaluation_task1 import voc_eval as voc_eval_task1
from dota_evaluation_task2 import voc_eval as voc_eval_task2
from nms.nms_poly import poly_gpu_nms
from nms.nms import gpu_nms

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
              'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
              'harbor', 'swimming-pool', 'helicopter']
## the thresh for nms when merge image, decrease the thresh of objects which has high aspect ratio
# use_differ_nms_thresh_per_cls = False
use_differ_nms_thresh_per_cls = True
if use_differ_nms_thresh_per_cls:
    r_threshold = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
               'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.05, 'plane': 0.3,
               'large-vehicle': 0.3, 'helicopter': 0.2, 'harbor': 0.2, 'ground-track-field': 0.3,
               'bridge': 0.2, 'basketball-court': 0.3, 'baseball-diamond': 0.3}
    h_threshold = {'roundabout': 0.35, 'tennis-court': 0.35, 'swimming-pool': 0.4, 'storage-tank': 0.3,
               'soccer-ball-field': 0.3, 'small-vehicle': 0.4, 'ship': 0.35, 'plane': 0.35,
               'large-vehicle': 0.4, 'helicopter': 0.4, 'harbor': 0.3, 'ground-track-field': 0.4,
               'bridge': 0.3, 'basketball-court': 0.4, 'baseball-diamond': 0.3}
else:
    nms_thresh = 0.3
    r_threshold = {c: nms_thresh for c in classnames}
    h_threshold = r_threshold


def nms_wrapper(task, device_id=0):
    def _nms(dets, thresh):
        dets = dets.astype(np.float32)
        if task == 1:
            return poly_gpu_nms(dets, thresh, device_id)
        else:
            return gpu_nms(dets, thresh, device_id)
    return _nms


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_colormap(n_colors):
    """ Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color extracted by the HSV colormap """

    import matplotlib.cm as cmx
    from matplotlib import colors

    color_norm = colors.Normalize(vmin=0, vmax=n_colors - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb(index):
        """ closure mapping index to color - range [0 255] for OpenCV """
        return 255 * np.array(scalar_map.to_rgba(index))

    return map_index_to_rgb


def draw_detection(im_name, detections, cmap):
    """ draw bounding boxes in the form (class, box, score) """

    im = cv2.imread(im_name)
    for j, name in enumerate(detections):
        if name == '__background__':
            continue
        # color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        color = cmap(j)
        dets = detections[name]
        for det in dets:
            bbox = det[:-1]
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # cv2.putText(im, '%s %.2f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
            #            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def draw_quadrangle_detection(im_name, detections, cmap):
    """
    visualize all detections in one image
    """
    # import random
    im = cv2.imread(im_name)
    # color_white = (255, 255, 255)
    for j, name in enumerate(detections):
        if name == '__background__':
            continue
        # color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        color = cmap(j)
        dets = detections[name]
        for det in dets:
            bbox = det[:-1]
            score = det[-1]
            bbox = map(int, bbox)
            for i in range(3):
                cv2.line(im, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                         thickness=2)
            cv2.line(im, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
            # cv2.putText(im, '%s %.2f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
            #            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def draw_merged(img_result, img_dir, out_dir, task=1):
    cmap = get_colormap(15)
    for imgname, dets in img_result.items():
        im_name = img_dir + '/' + imgname + '.png'
        if task == 1:
            im = draw_quadrangle_detection(im_name, dets, cmap)
        else:
            im = draw_detection(im_name, dets, cmap)
        out_name = out_dir + '/' + imgname + '.png'
        cv2.imwrite(out_name, im)


def eval_merged_result(model_output_path, dota_home=r'../DOTA', task=1):
    if task == 1:
        voc_eval = voc_eval_task1
    else:
        voc_eval = voc_eval_task2
    detpath = os.path.join(model_output_path, 'merge_results/Task{}'.format(task) + '_{:s}.txt')
    # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    annopath = dota_home + '/val/labelTxt/{:s}.txt'
    imagesetfile = dota_home + '/valset.txt'

    if 'c6' in model_output_path:
        classnames = ['plane', 'ship', 'storage-tank', 'harbor', 'bridge', 'large-vehicle']

    classaps = []
    aps = []
    map = 0
    for classname in classnames:
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        # print '%s: %.2f'%(classname,ap*100)
        classaps.append(ap)
        aps.append('%.2f' % (ap * 100))

        # show p-r curve of each category
        show_pr = False
        if show_pr:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.plot(rec, prec, lw=2, # plt.plot(rec, prec)
                     label='Precision-recall curve of class {} (area = {:.4f})'.format(classname, ap))
            # plt.show()
            plt.savefig('PR-curv-%s.png' % classname)

    map = map / len(classnames)
    return map, aps, classnames


def start_eval(model_output_path, task=1, draw=False, dota_home=r'../DOTA'):
    '''
    merge and eval, the detection result files by class locate in model_output_path
    model_output_path = r'../DCR/output/rcnn/DOTA_quadrangle/DOTA_quadrangle/val'
    the dota_home include /val/labelTxt/ & valset.txt(which include big image id)
    '''
    sys.stdout.write('merging...')
    sys.stdout.flush()
    img_dir = dota_home + '/val/images'
    dstpath = model_output_path + '/merge_results'
    out_img_dir = model_output_path + '/vis'
    mkdir(dstpath)
    mkdir(out_img_dir)
    srcpath = model_output_path + '/test_results_by_class_task{}'.format(task)
    nms_thresh_by_cls = r_threshold if task == 1 else h_threshold
    img_result = merge_result.mergebase(srcpath, dstpath, nms=nms_wrapper(task),
                                        nms_thresh_by_cls=nms_thresh_by_cls)

    if draw:
        sys.stdout.write('\rstart draw...')
        sys.stdout.flush()
        draw_merged(img_result, img_dir, out_img_dir, task=task)
    sys.stdout.write('\rstart eval...')
    sys.stdout.flush()

    map, aps, classnames = eval_merged_result(model_output_path, dota_home=dota_home, task=task)
    classnames = ["%-12s"%c[:12] for c in classnames]
    aps = ["%-12s"%c for c in aps]
    info_str = 'Task{} mAP = {}'.format(task, '%.2f' % (map * 100))
    info_str2 = '\n' + '\t'.join(classnames) + '\n' + '\t'.join(aps) + '\n'
    sys.stdout.write('\rTask{} mAP = \x1b[0;32;40m{}\x1b[0m'.format(task, '%.2f' % (map * 100)))
    sys.stdout.write(info_str2)
    sys.stdout.flush()

    return info_str + info_str2


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: eval model_output_path [task 1/2, default 1] [draw, default false]'
        sys.exit(0)

    model_output_path = sys.argv[1]
    if len(sys.argv) > 2:
        task = int(sys.argv[2])
    else:
        task = 1
    if len(sys.argv) > 3:
        draw = strtobool(sys.argv[3])
    else:
        draw = False

    start_eval(model_output_path, task=task, draw=draw)