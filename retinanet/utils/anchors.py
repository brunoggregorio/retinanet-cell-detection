"""!
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications copyright (C) 2019 Bruno Gregorio - BIPG

    https://brunoggregorio.github.io \n
    https://www.bipgroup.dc.ufscar.br
"""

import numpy as np
import keras

from .compute_overlap import compute_overlap


class AnchorParameters:
    """!@brief
    The parameteres that define how anchors are generated.

    @param sizes   : List of sizes to use. Each size corresponds to one feature level.
    @param strides : List of strides to use. Each stride correspond to one feature level.
    @param ratios  : List of ratios to use per location in a feature map.
    @param scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""!@brief
    The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes   = [8, 16, 32, 64], # [32, 64, 128, 256, 512]
    strides = [4, 8, 16, 32], # [ 8, 16,  32,  64, 128]
    ratios  = np.array([0.5, 1, 1.5], keras.backend.floatx()),  # [0.5, 1, 2]
    scales  = np.array([2 ** 0, 2 ** (1.0/4.0), 2 ** (2.0/4.0), 2 ** (3.0/4.0)], keras.backend.floatx()), # [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
)


def anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """!@brief
    Generate anchor targets for bbox detection.

    @param anchors           : np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
    @param image_group       : List of BGR images.
    @param annotations_group : List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
    @param num_classes       : Number of classes to predict.
    @param mask_shape        : If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
    @param negative_overlap  : IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
    @param positive_overlap  : IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    @returns
        labels_batch     : batch that contains labels & anchor states (np.array
                           of shape (batch_size, N, num_classes + 1), where N is
                           the number of anchors for an image and the last column
                           defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch : batch that contains bounding-box regression targets
                           for an image & anchor states (np.array of shape
                           (batch_size, N, 4 + 1), where N is the number of anchors
                           for an image, the first 4 columns define regression
                           targets for (x1, y1, x2, y2) and the last column defines
                           anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)

    regression_batch  = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch      = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1]       = -1
            labels_batch[index, positive_indices, -1]     = 1

            regression_batch[index, ignore_indices, -1]   = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1]     = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch


def compute_gt_annotations(
    anchors,
    annotations,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """!@brief
    Obtain indices of gt annotations with the greatest overlap.

    @param anchors          : np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
    @param annotations      : np.array of shape (N, 5) for (x1, y1, x2, y2, label).
    @param negative_overlap : IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
    @param positive_overlap : IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    @returns
        positive_indices     : indices of positive anchors
        ignore_indices       : indices of ignored anchors
        argmax_overlaps_inds : ordered overlaps indices
    """

    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """!@brief
    Compute layer shapes given input image shape and the model.

    @param image_shape : The shape of the image.
    @param model       : The model to use for computing how the image shape is
                         transformed in the pyramid.

    @return
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    """!@brief
    Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """!@brief
    Guess shapes based on pyramid levels.

    @param image_shape    : The shape of the image.
    @param pyramid_levels : A list of what pyramid levels are used.

    @return
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    anchor_params=None,
    shapes_callback=None,
):
    """!@brief
    Generate anchors for a given shape.

    @param image_shape     : The shape of the image.
    @param pyramid_levels  : List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
    @param anchor_params   : Struct containing anchor parameters. If None, default values are used.
    @param shapes_callback : Function to call for getting the shape of the image at different pyramid levels.

    @return
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [2, 3, 4, 5]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    """!@brief
    Produce shifted anchors based on shape of the map and stride size.

    @param shape  : Shape to shift the anchors over.
    @param stride : Stride to shift the anchors with over the shape.
    @param anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """!@brief
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """!@brief
    Compute bounding-box regression targets for an image.
    """

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets
