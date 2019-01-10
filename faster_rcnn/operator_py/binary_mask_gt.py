# -------------------------
# Binary Mask from Ground truth boxes
# write by fyk
# -------------------------
import mxnet as mx
import numpy as np
import cv2

class BinaryMaskGTOperator(mx.operator.CustomOp):
    def __init__(self, spatial_scale):
        super(BinaryMaskGTOperator, self).__init__()
        self._spatial_scale = spatial_scale


    def forward(self, is_train, req, in_data, out_data, aux):

        bin_mask_pred = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        # create array shape like bin_mask_pred, scale the gt_boxes and draw the boxes in bin_mask_gt
        bin_mask_gt = np.zeros_like(bin_mask_pred[0, 0, :, :], np.uint8)
        assert bin_mask_gt.ndim == 2
        self.fill_poly(bin_mask_gt, gt_boxes, self._spatial_scale)

        self.assign(out_data[0], req[0], bin_mask_gt.reshape(bin_mask_pred.shape))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


    def fill_poly(self, img, gt_boxes, spatial_scale):
        areas = []
        # [x1, y1, x2, y2, x3, y3, x4, y4, class] point in clockwise
        for b in gt_boxes:
            a = np.array([ [round(b[0] * spatial_scale), round(b[1] * spatial_scale)], [round(b[2] * spatial_scale), round(b[3] * spatial_scale)],
                           [round(b[4] * spatial_scale), round(b[5] * spatial_scale)], [round(b[6] * spatial_scale), round(b[7] * spatial_scale)] ],
                         np.int32) # numpy type for fillPoly must be integer
            areas.append(a)

        cv2.fillPoly(img, areas, color=1)
        # triangle = np.array([[1, 3], [4, 8], [1, 9]], np.uint32)
        # faster than fillPoly (which is both for convex poly and complex poly), but only accept one poly
        # cv2.fillConvexPoly(a, triangle, 1)


@mx.operator.register('BinaryMaskGt')
class BinaryMaskGTProp(mx.operator.CustomOpProp):
    def __init__(self, spatial_scale='0.0625'):
        super(BinaryMaskGTProp, self).__init__(need_top_grad=False)
        self._spatial_scale = float(spatial_scale) # such as 1/16.

    def list_arguments(self):
        return ['bin_mask_pred', 'gt_boxes']

    def list_outputs(self):
        return ['bin_mask_gt']

    def infer_shape(self, in_shape):
        bin_mask_pred_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        assert gt_boxes_shape[1] == 9, 'this layer is write for quadrangle 8 point'

        return [bin_mask_pred_shape, gt_boxes_shape], [bin_mask_pred_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BinaryMaskGTOperator(self._spatial_scale)


    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []