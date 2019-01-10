import os
import numpy as np
import re
# import time

import dota_utils as util

def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        outdets = []
        for index in keep:
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict

def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergebase(srcpath, dstpath, nms, nms_thresh_by_cls):
    """
    merge small image result to big one
    :param srcpath: result files before merge and nms
    :param dstpath: result files after merge and nms
    :param nms: polygon nms or horizon bbox nms
    :param nms_thresh_by_cls: per thresh for each class
    :return: img_result for draw and display
    """
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    img_result = {}
    filelist = util.GetFileFromThisRootDir(srcpath)
    for fullname in filelist:
        name = util.custombasename(fullname)
        dstname = os.path.join(dstpath, name + '.txt')
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            lines = f_in.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                subname = splitline[0]
                splitname = subname.split('__')
                oriname = splitname[0]
                pattern1 = re.compile(r'__\d+___\d+')
                x_y = re.findall(pattern1, subname)
                x_y_2 = re.findall(r'\d+', x_y[0])
                x, y = int(x_y_2[0]), int(x_y_2[1])

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
                rate = re.findall(pattern2, subname)[0]
                confidence = splitline[1]
                poly = list(map(float, splitline[2:]))
                origpoly = poly2origpoly(poly, x, y, rate)
                det = origpoly
                det.append(confidence)
                det = list(map(float, det))
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []
                nameboxdict[oriname].append(det)
            cls_name = name.split('_')[1]
            nms_thresh = nms_thresh_by_cls[cls_name]
            nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    if (imgname not in img_result):
                        img_result[imgname] = {}
                    if (name not in img_result[imgname]):
                        img_result[imgname][name] = []
                    for det in nameboxnmsdict[imgname]:
                        img_result[imgname][name].append(det)
                        confidence = det[-1]
                        bbox = det[0:-1]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                        f_out.write(outline + '\n')
    return img_result
