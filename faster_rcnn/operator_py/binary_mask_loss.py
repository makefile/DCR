# -------------------------
# Binary Mask loss (binary_cross_entropy)
# todo it is a little hard to write the code for backward,
# the better way is change this layer to bin_mask_gt layer
# and write loss in main symbol
# -------------------------
import mxnet as mx
import numpy as np
import cv2

class BinaryMaskLossOperator(mx.operator.CustomOp):
    def __init__(self, spatial_scale):
        super(BinaryMaskLossOperator, self).__init__()
        self._spatial_scale = spatial_scale


    def forward(self, is_train, req, in_data, out_data, aux):

        bin_mask_pred = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        # todo create array shape like bin_mask_pred, scale the gt_boxes and draw the boxes in bin_mask_gt
        bin_mask_gt = np.zeros_like(bin_mask_pred, np.uint8)
        self.fill_poly(bin_mask_gt, gt_boxes, self._spatial_scale)
        ce_loss = -(np.log(bin_mask_pred + 1e-12) * bin_mask_gt +
                    np.log(1. - bin_mask_pred + 1e-12) * (1. - bin_mask_gt))

        self.assign(out_data[0], req[0], mx.nd.array(ce_loss))
        self.assign(out_data[1], req[1], bin_mask_gt)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # todo
        a = in_data[0].asnumpy() # bin_mask_pred
        y = out_data[1].asnumpy() # bin_mask_gt

        grad = (a-y)/a/(1-a)

        self.assign(in_grad[0], req[0], mx.nd.array(grad))


    def fill_poly(self, img, gt_boxes, spatial_scale):
        areas = []
        for b in gt_boxes:
            a = np.array([ [b[0] * spatial_scale, b[1] * spatial_scale], [b[2] * spatial_scale, b[3] * spatial_scale],
                           [b[4] * spatial_scale, b[5] * spatial_scale], [b[6] * spatial_scale, b[7] * spatial_scale] ], np.uint32)
            areas.append(a)

        cv2.fillPoly(img, areas, color = 1)
        # triangle = np.array([[1, 3], [4, 8], [1, 9]], np.uint32)
        # faster than fillPoly (which is both for convex poly and complex poly), but only accept one poly
        # cv2.fillConvexPoly(a, triangle, 1)


@mx.operator.register('binary_mask_loss')
class BinaryMaskLossProp(mx.operator.CustomOpProp):
    def __init__(self, spatial_scale='0.0625'):
        super(BinaryMaskLossProp, self).__init__(need_top_grad=False)
        self._spatial_scale = float(spatial_scale) # such as 1/16.

    def list_arguments(self):
        return ['bin_mask_pred', 'gt_boxes']

    def list_outputs(self):
        return ['bin_mask_loss', 'bin_mask_gt']

    def infer_shape(self, in_shape):
        bin_mask_pred_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        assert gt_boxes_shape[1] == 9, 'this layer is write for quadrangle 8 point'
        # assert bin_mask_pred_shape == mask_gt_shape, 'size does not match: mask_pred <> mask_gt'
        output_shape = [1, ] # loss shape

        return [bin_mask_pred_shape, gt_boxes_shape], [output_shape, bin_mask_pred_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BinaryMaskLossOperator(self._spatial_scale)


    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []