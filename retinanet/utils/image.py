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

from __future__ import division
import numpy as np
import cv2
import math
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from .transform import change_transform_origin


class TransformParameters:
    """!@brief
    Struct holding parameters determining how to apply a transformation to an image.

    @param fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
    @param interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
    @param cval:                  Fill value to use with fill_mode='constant'
    @param relative_translation:  If true (the default), interpret translation as a factor of the image size.
                                  If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'constant', #'nearest',
        interpolation        = 'cubic',    #'linear',
        cval                 = 0,          #(127,127,127),
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def read_image_bgr(path):
    """!@brief
    Read an image in BGR format.

    @param path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image(x, mode='caffe'):
    """!@brief
    Preprocess an image by subtracting the ImageNet mean.

    @param x    : np.array of shape (None, None, 3) or (3, None, None).
    @param mode : One of "caffe" or "tf".
                    - caffe: will zero-center each color channel with
                        respect to the ImageNet dataset, without scaling.
                    - tf: will scale pixels between -1 and 1, sample-wise.

    @returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939    # B
        x[..., 1] -= 116.779    # G
        x[..., 2] -= 123.68     # R << I think...

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """!@brief
    Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of
    the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


def adjust_deformable_for_image(shape,
                                alpha=(1,200),
                                sigma=(4,7),
                                random_state=None):
    """!@brief
    Get a smooth elastic transformation array.

    @see
        Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis",
           in Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

    @param shape : The image shape to transform.
    @param alpha : Scaling factor that controls the intensity of the deformation.
    @param sigma : Gaussian standard deviation (in pixels).
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    alpha = random_state.uniform(alpha[0], alpha[1])
    sigma = random_state.uniform(sigma[0], sigma[1])

    #-------------------------------------------
    # # Print deformable params
    # print("Deformable param: alpha = ", alpha)
    # print("Deformable param: sigma = ", sigma)
    #-------------------------------------------

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    if len(shape) == 3:
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    elif len(shape) == 2:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return indices


def apply_transform(matrix, image, params):
    """!@brief
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is
    moved to transform * (x, y) in the generated image.

    Mathematically speaking, that means that the matrix is a transformation from
    the transformed image space to the original image space.

    @param matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
    @param image:  The image to transform.
    @param params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )

    return output


def apply_deformable_transform(image, transform):
    """!@brief
    Apply a smooth elastic transformation to the image.

    @see
        Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis",
           in Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

    @param image     : The image to be transformated.
    @param transform : The transformation array.
    """
    distorted_image = map_coordinates(image, transform, order=1, mode='constant')

    #-------------------------------------------
    # # Print arrays' shapes
    # print("Image shape: ", image.shape) # (244, 360, 3)
    # print("Transform shape: ", np.shape(transform)) # (3, 263520)
    # print("Distorted image shape: ", distorted_image.shape) # (263520, 1)
    #-------------------------------------------

    return distorted_image.reshape(image.shape)


def apply_motion_blur(image, kernel=None):
    """!@brief
    Apply a PSF blur to the image using different kernel types.

    @param image  : The image to apply the blur.
    @param kernel : The kernel to be used as blurring PSF.

    @return
        An image blurred.
    """
    if kernel is None:
        kernel = createPSFKernel()

    # Apply the convolution with defined kernel
    image = cv2.filter2D(image, -1, kernel)

    return image


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """!@brief
    Compute an image scale such that the image size is constrained to min_side and max_side.

    @param min_side: The image's min side will be equal to min_side after resizing.
    @param max_side: If after resizing the image's max side is above max_side,
                     resize until the max side is equal to max_side.

    @return
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """!@brief
    Resize an image such that the size is constrained to min_side and max_side.

    @param min_side: The image's min side will be equal to min_side after resizing.
    @param max_side: If after resizing the image's max side is above max_side,
                     resize until the max side is equal to max_side.

    @return
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
