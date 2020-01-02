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
import cv2
from scipy.ndimage.interpolation import map_coordinates
from .photometric_ops import (
    ConvertColor,
    ConvertDataType,
    ConvertTo3Channels,
    RandomBrightness,
    RandomContrast,
    RandomHue,
    RandomSaturation,
    RandomChannelSwap
)

DEFAULT_PRNG = np.random


def colvec(*args):
    """!@brief
    Create a numpy array representing a column vector. """
    return np.array([args]).T


def photometric_distortions(image, labels):
    """!@brief
    Performs the photometric distortions defined by the `train_transform_param` instructions
    of the original Caffe implementation of SSD.
    """

    convert_RGB_to_HSV    = ConvertColor(current='RGB', to='HSV')
    convert_HSV_to_RGB    = ConvertColor(current='HSV', to='RGB')
    convert_to_float32    = ConvertDataType(to='float32')
    convert_to_uint8      = ConvertDataType(to='uint8')
    convert_to_3_channels = ConvertTo3Channels()
    random_brightness     = RandomBrightness(lower=-16, upper=16, prob=0.5)
    random_contrast       = RandomContrast(lower=0.8, upper=1.5, prob=0.5)
    random_saturation     = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
    random_hue            = RandomHue(max_delta=18, prob=0.5)
    random_channel_swap   = RandomChannelSwap(prob=0.5)

    sequence1 = [convert_to_3_channels,
                  convert_to_float32,
                  random_brightness,
                  random_contrast,
                  convert_to_uint8,
                  convert_RGB_to_HSV,
                  convert_to_float32,
                  random_saturation,
                  random_hue,
                  convert_to_uint8,
                  convert_HSV_to_RGB,
                  random_channel_swap]

    sequence2 = [convert_to_3_channels,
                  convert_to_float32,
                  random_brightness,
                  convert_to_uint8,
                  convert_RGB_to_HSV,
                  convert_to_float32,
                  random_saturation,
                  random_hue,
                  convert_to_uint8,
                  convert_HSV_to_RGB,
                  convert_to_float32,
                  random_contrast,
                  convert_to_uint8,
                  random_channel_swap]

    # Choose sequence 1 with probability 0.5.
    if np.random.choice(2):
        for transform in sequence1:
            image, labels = transform(image, labels)
        return image, labels
    # Choose sequence 2 with probability 0.5.
    else:
        for transform in sequence2:
            image, labels = transform(image, labels)
        return image, labels


def transform_aabb(transform, aabb):
    """!@brief
    Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying
    the given transformation.

    @param transform : The transformation to apply.
    @param x1        : The minimum x value of the AABB.
    @param y1        : The minimum y value of the AABB.
    @param x2        : The maximum x value of the AABB.
    @param y2        : The maximum y value of the AABB.

    @return
        The new AABB as tuple (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def deformable_transform_aabb(shape, transform, aabb):
    """!@brief
    Apply a deformable transformation to the bounding boxes.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying
    the given transformation.

    @note
        In OPENCV:                      Numpy:
            (0,0) --- x --- > cols          np.shape(rows,cols,ndim)
              |                                        y ,  x
              y / rows
              |
              v

    @param shape:     The shape of the input image.
    @param transform: The transformation to apply.
    @param x1:        The minimum x value of the AABB.
    @param y1:        The minimum y value of the AABB.
    @param x2:        The maximum x value of the AABB.
    @param y2:        The maximum y value of the AABB.

    @return
        The new AABB as tuple (x1, y1, x2, y2).
    """
    # Get number of image rows and cols
    rows = shape[0] # y-axis in OpenCV
    cols = shape[1] # x-axis in OpenCV

    # Input bbox points
    x1, y1, x2, y2 = aabb.astype(int)

    # Get bbox size
    bbox_w = x2-x1+1
    bbox_h = y2-y1+1

    # Set additional margin
    m = 5

    # Set ROI
    #   (11,21)-----
    #      |       |
    #      -----(12,22)
    roi_11 = np.clip(x1-m-1, 0, cols-1)
    roi_21 = np.clip(y1-m-1, 0, rows-1)
    roi_12 = np.clip(x2+m,   0, cols-1)
    roi_22 = np.clip(y2+m,   0, rows-1)

    roi_transform = []
    for i in range(len(transform)):
        # Transform to image shape and applies a ROI
        roi_transform.append(transform[i].reshape(shape)[roi_21:roi_22, roi_11:roi_12])

    # Get ROI shape
    roi_shape = roi_transform[0].shape

    # Compute indices for map_coordinates
    roi_indices = np.reshape(roi_transform[0], (-1, 1)), np.reshape(roi_transform[1], (-1, 1)), np.reshape(roi_transform[2], (-1, 1))

    # Create temporary image with bbox
    tmp_img = np.zeros(shape)
    cv2.rectangle(tmp_img,(x1,y1),(x2,y2),(255,255,255),1)

    # scipy.ndimage.interpolation.map_coordinates:
    #
    # Map the input array to new coordinates by interpolation.
    #
    # The array of coordinates is used to find, for each point in the output,
    # the corresponding coordinates in the input. The value of the input at
    # those coordinates is determined by spline interpolation of the
    # requested order.
    #
    # The shape of the output is derived from that of the coordinate
    # array by dropping the first axis. The values of the array along
    # the first axis are the coordinates in the input array at which the
    # output value is found.
    distorted = map_coordinates(tmp_img, roi_indices, order=1, mode='constant', cval=0.0).reshape(roi_shape)

    # Find extreme points in ROI distorted image
    A = np.nonzero(distorted>0)
    x1_roi = min(A[1])
    y1_roi = min(A[0])
    x2_roi = max(A[1])
    y2_roi = max(A[0])

    # Adjust points according to image coordinates
    x1_ = np.clip(roi_11 + x1_roi, 0, cols-1)
    y1_ = np.clip(roi_21 + y1_roi, 0, rows-1)
    x2_ = np.clip(roi_11 + x2_roi, 0, cols-1)
    y2_ = np.clip(roi_21 + y2_roi, 0, rows-1)

    #-------------------------------------------
    # # Draw the findings
    # tmp_img[roi_21:roi_22, roi_11:roi_12] = distorted
    # cv2.circle(tmp_img, (x1_,y1_), 3, (0,255,0), 1)
    # cv2.circle(tmp_img, (x2_,y2_), 3, (0,255,0), 1)
    # print("Top left: ({}, {}) | Bottom right: ({}, {})".format(x1, y1, x2, y2))
    # print("ROI Top left: ({}, {}) | ROI Bottom right: ({}, {})".format(x1_roi, y1_roi, x2_roi, y2_roi))
    # print("New top left: ({}, {}) | New bottom right: ({}, {})".format(x1_, y1_, x2_, y2_))
    #
    # print("Temp image shape: ", tmp_img.shape)
    # cv2.imshow("Temp Image", tmp_img)
    # cv2.waitKey(0)
    #-------------------------------------------

    return [x1_, y1_, x2_, y2_]


# Delete backup:
def deformable_transform_aabb_bkp(shape, transform, aabb):
    """!@brief
    Apply a deformable transformation to the bounding boxes.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    @param shape:     The shape of the input image.
    @param transform: The transformation to apply.
    @param x1:        The minimum x value of the AABB.
    @param y1:        The minimum y value of the AABB.
    @param x2:        The maximum x value of the AABB.
    @param y2:        The maximum y value of the AABB.

    @todo
        Find a better way to transform these values.

    @return
        The new AABB as tuple (x1, y1, x2, y2).
    """
    # Input bbox points
    x1, y1, x2, y2 = aabb.astype(int)

    # Create image with bbox
    tmp_img = np.zeros(shape)
    cv2.rectangle(tmp_img,(x1,y1),(x2,y2),(255,255,255),1)

    # scipy.ndimage.interpolation.map_coordinates:
    #
    #     Map the input array to new coordinates by interpolation.
    #
    #     The array of coordinates is used to find, for each point in the output,
    #     the corresponding coordinates in the input. The value of the input at
    #     those coordinates is determined by spline interpolation of the
    #     requested order.
    #
    #     The shape of the output is derived from that of the coordinate
    #     array by dropping the first axis. The values of the array along
    #     the first axis are the coordinates in the input array at which the
    #     output value is found.
    distorted = map_coordinates(tmp_img, transform, order=1, mode='reflect').reshape(shape)

    A = np.nonzero(distorted>0)
    x1_ = min(A[1])
    y1_ = min(A[0])
    x2_ = max(A[1])
    y2_ = max(A[0])

    #-------------------------------------------
    # # Draw the findings
    # cv2.circle(distorted, (x1_,y1_), 3, (0,255,0), 1)
    # cv2.circle(distorted, (x2_,y2_), 3, (0,255,0), 1)
    # print("Top left: [",x1,",",y1,"] | Bottom right: [",x2,",",y2,"]")
    # print("New top left: [",x1_,",",y1_,"] | New bottom right: [",x2_,",",y2_,"]")
    #
    # print("Temp image shape: ", distorted.shape)
    # cv2.imshow("Temp Image", distorted)
    # cv2.waitKey(0)
    #-------------------------------------------

    return [x1_, y1_, x2_, y2_]


def motion_distortion_aabb(shape, kernel, aabb):
    """!@brief
    Adjust the bounding box points according to the blur applied.

    @param shape  : The shape of the image being blurred.
    @param kernel : The kernel used to blur the image.
    @param aabb   : Coordinates of the bounding box points.

    @return
        The adjusted bounding box coordinates.
    """
    # Get number of image rows and cols
    rows = shape[0] # y-axis in OpenCV
    cols = shape[1] # x-axis in OpenCV

    # Get point values
    x1, y1, x2, y2 = aabb

    # Find the value to adjust bboxes
    k_delta = (kernel.shape[0]-1)/4

    # Adjust points aabb
    x1_ = np.clip(x1-k_delta, 0, cols-1)
    y1_ = np.clip(y1-k_delta, 0, rows-1)
    x2_ = np.clip(x2+k_delta, 0, cols-1)
    y2_ = np.clip(y2+k_delta, 0, rows-1)

    return [x1_, y1_, x2_, y2_]


def _random_vector(min, max, prng=DEFAULT_PRNG):
    """!@brief
    Construct a random vector between min and max.

    @param min : The minimum value for each component.
    @param max : The maximum value for each component.

    @return
        The vector created.
    """
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def rotation(angle):
    """!@brief
    Construct a homogeneous 2D rotation matrix.

    @param angle : The angle in radians.

    @return
        The rotation matrix as 3 by 3 numpy array.
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(min, max, prng=DEFAULT_PRNG):
    """!@brief
    Construct a random rotation between -max and max.

    @param min  : A scalar for the minimum absolute angle in radians.
    @param max  : A scalar for the maximum absolute angle in radians.
    @param prng : The pseudo-random number generator to use.

    @return
        A homogeneous 3 by 3 rotation matrix.
    """
    return rotation(prng.uniform(min, max))


def translation(translation):
    """!@brief
    Construct a homogeneous 2D translation matrix.

    @param translation : The translation 2D vector.

    @return
        The translation matrix as 3 by 3 numpy array.
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def random_translation(min, max, prng=DEFAULT_PRNG):
    """!@brief
    Construct a random 2D translation between min and max.

    @param min  : A 2D vector with the minimum translation for each dimension.
    @param max  : A 2D vector with the maximum translation for each dimension.
    @param prng : The pseudo-random number generator to use.

    @return
        A homogeneous 3 by 3 translation matrix.
    """
    return translation(_random_vector(min, max, prng))


def shear(angle):
    """!@brief
    Construct a homogeneous 2D shear matrix.

    @param angle : The shear angle in radians.

    @return
        The shear matrix as 3 by 3 numpy array.
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0,  np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_shear(min, max, prng=DEFAULT_PRNG):
    """!@brief
    Construct a random 2D shear matrix with shear angle between -max and max.

    @param min  : The minimum shear angle in radians.
    @param max  : The maximum shear angle in radians.
    @param prng : The pseudo-random number generator to use.

    @return
        A homogeneous 3 by 3 shear matrix.
    """
    return shear(prng.uniform(min, max))


def scaling(factor):
    """ Construct a homogeneous 2D scaling matrix.
    Args
        factor: a 2D vector for X and Y scaling
    Returns
        the zoom matrix as 3 by 3 numpy array
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng=DEFAULT_PRNG):
    """!@brief
    Construct a random 2D scale matrix between -max and max.

    @param min  : A 2D vector containing the minimum scaling factor for X and Y.
    @param min  : A 2D vector containing The maximum scaling factor for X and Y.
    @param prng : The pseudo-random number generator to use.

    @return
        A homogeneous 3 by 3 scaling matrix.
    """
    factor = _random_vector(min, max, prng)

    # To scale x and y by the same factor or
    # at least a close one
    ratio = factor[0]/factor[1]
    if ratio < 0.9 or ratio > 1.1:
        factor[0] = factor[1]

    return scaling(factor)


def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    """!@brief
    Construct a transformation randomly containing X/Y flips (or not).

    @param flip_x_chance : The chance that the result will contain a flip along the X axis.
    @param flip_y_chance : The chance that the result will contain a flip along the Y axis.
    @param prng          : The pseudo-random number generator to use.

    @return
        A homogeneous 3 by 3 transformation matrix.
    """
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))


def change_transform_origin(transform, center):
    """!@brief
    Create a new transform representing the same transformation,
    only with the origin of the linear part changed.

    @param transform : The transformation matrix.
    @param center    : The new origin of the transformation.

    @return
        Translate(center) * transform * translate(-center).
    """
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


def random_transform(min_rotation=0,
                     max_rotation=0,
                     min_translation=(0, 0),
                     max_translation=(0, 0),
                     min_shear=0,
                     max_shear=0,
                     min_scaling=(1, 1),
                     max_scaling=(1, 1),
                     flip_x_chance=0,
                     flip_y_chance=0,
                     prng=DEFAULT_PRNG):
    """!@brief
    Create a random transformation.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    @param min_rotation    : The minimum rotation in radians for the transform as scalar.
    @param max_rotation    : The maximum rotation in radians for the transform as scalar.
    @param min_translation : The minimum translation for the transform as 2D column vector.
    @param max_translation : The maximum translation for the transform as 2D column vector.
    @param min_shear       : The minimum shear angle for the transform in radians.
    @param max_shear       : The maximum shear angle for the transform in radians.
    @param min_scaling     : The minimum scaling for the transform as 2D column vector.
    @param max_scaling     : The maximum scaling for the transform as 2D column vector.
    @param flip_x_chance   : The chance (0 to 1) that a transform will contain a flip along X direction.
    @param flip_y_chance   : The chance (0 to 1) that a transform will contain a flip along Y direction.
    @param prng            : The pseudo-random number generator to use.
    """
    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, prng),
        random_translation(min_translation, max_translation, prng),
        random_shear(min_shear, max_shear, prng),
        random_scaling(min_scaling, max_scaling, prng),
        random_flip(flip_x_chance, flip_y_chance, prng)
    ])


def random_transform_generator(prng=None, **kwargs):
    """!@brief
    Create a random transform generator.

    Uses a dedicated, newly created, properly seeded PRNG by default instead of the global DEFAULT_PRNG.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    @param min_rotation    : The minimum rotation in radians for the transform as scalar.
    @param max_rotation    : The maximum rotation in radians for the transform as scalar.
    @param min_translation : The minimum translation for the transform as 2D column vector.
    @param max_translation : The maximum translation for the transform as 2D column vector.
    @param min_shear       : The minimum shear angle for the transform in radians.
    @param max_shear       : The maximum shear angle for the transform in radians.
    @param min_scaling     : The minimum scaling for the transform as 2D column vector.
    @param max_scaling     : The maximum scaling for the transform as 2D column vector.
    @param flip_x_chance   : The chance (0 to 1) that a transform will contain a flip along X direction.
    @param flip_y_chance   : The chance (0 to 1) that a transform will contain a flip along Y direction.
    @param prng            : The pseudo-random number generator to use.
    """

    if prng is None:
        # RandomState automatically seeds using the best available method.
        prng = np.random.RandomState()

#     print('- Generator initiated - ')
#     idx = 0
    while True:
        yield random_transform(prng=prng, **kwargs)
#         print('Generator yielded a batch %d' % idx)
#         idx += 1
