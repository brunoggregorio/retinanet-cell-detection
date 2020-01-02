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
import random
import warnings

import cv2
import keras
import multiprocessing as mp

from utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from utils.config import parse_anchor_parameters
from utils.point_spread_functions import createPSFKernel

from utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    adjust_deformable_for_image,
    apply_transform,
    apply_deformable_transform,
    apply_motion_blur,
    preprocess_image,
    resize_image,
)
from utils.transform import (
    transform_aabb,
    deformable_transform_aabb,
    motion_distortion_aabb,
    photometric_distortions
)


class Generator(keras.utils.Sequence):
    """!@brief
    Abstract generator class.
    """

    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        photometric=True,
        motion=True,
        deformable=True,
        alpha=(1,200),
        sigma=(8,10),
        config=None
    ):
        """!@brief
        Initialize Generator object.

        @param transform_generator    : A generator used to randomly transform
                                        images and annotations.
        @param batch_size             : The size of the batches to generate.
        @param group_method           : Determines how images are grouped together
                                        (defaults to 'ratio', one of ('none',
                                        'random', 'ratio')).
        @param shuffle_groups         : If True, shuffles the groups each epoch.
        @param image_min_side         : After resizing the minimum side of an image
                                        is equal to image_min_side.
        @param image_max_side         : If after resizing the maximum side is
                                        larger than image_max_side, scales down
                                        further so that the max side is equal to
                                        image_max_side.
        @param transform_parameters   : The transform parameters used for data
                                        augmentation.
        @param compute_anchor_targets : Function handler for computing the targets
                                        of anchors for an image and its annotations.
        @param compute_shapes         : Function handler for computing the shapes
                                        of the pyramid for a given input.
        @param preprocess_image       : Function handler for preprocessing an image
                                        (scaling / normalizing) for passing through
                                        a network.
        """
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.photometric            = photometric
        self.motion                 = motion
        self.deformable             = deformable
        self.alpha                  = alpha
        self.sigma                  = sigma
        self.config                 = config

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """!@brief
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """!@brief
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """!@brief
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """!@brief
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """!@brief
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """!@brief
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """!@brief
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """!@brief
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def get_image_name(self, image_index):
        """!@brief
        Get image file name at the image_index.
        """
        raise NotImplementedError('get_image_name method not implemented')

    def load_annotations(self, image_index):
        """!@brief
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """!@brief
        Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """!@brief
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """!@brief
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self,
                                     image,
                                     annotations,
                                     transform=None):
        """!@brief
        Randomly transforms image and annotation.
        """

        #-------------------------------------------
        # # Draw grid lines for print/debug purposes
        # grid_size = 20
        # for i in range(0, image.shape[1], grid_size):
        #     cv2.line(image, (i, 0), (i, image.shape[0]), color=(50,50,50))
        # for j in range(0, image.shape[0], grid_size):
        #     cv2.line(image, (0, j), (image.shape[1], j), color=(50,50,50))
        #
        # # Draw the annotations
        # from utils.visualization import draw_annotations
        # draw_annotations(image,
        #                  annotations,
        #                  color=(0, 0, 255))
        #
        # # Show input image
        # cv2.imshow("Input", image)
        #-------------------------------------------

        # Randomly apply photometric distortions
        if self.photometric:
            # Apply photometric distortion to the image
            image, annotations = photometric_distortions(image, annotations)

        # Randomly apply point-spread-function (PSF)
        # blurring to simulate animal motion
        if self.motion:
            # Create a random PSF kernel
            kernel = createPSFKernel(image.shape)

            # Kernel size can be small enough (< 3)
            # to not apply the motion blur.
            if kernel is not None:
                # Apply PSF blur to the image
                image = apply_motion_blur(image, kernel)

                # Adjust bounding boxes accordingly
                annotations['bboxes'] = annotations['bboxes'].copy()
                for index in range(annotations['bboxes'].shape[0]):
                    annotations['bboxes'][index, :] = motion_distortion_aabb(image.shape, kernel, annotations['bboxes'][index, :])

                #-------------------------------------------
                # # Draw the annotations
                # draw_annotations(image,
                #                  annotations,
                #                  color=(255, 0, 0))
                # cv2.imshow("PSF_Blur", image)
                #-------------------------------------------

        # Randomly apply smooth elastic deformation to the image
        if self.deformable:
            # Get deformable array
            deformable_transform = adjust_deformable_for_image(image.shape, self.alpha, self.sigma)

            # Apply transformation to the image
            image = apply_deformable_transform(image, deformable_transform)

            # Transform the bounding boxes in the annotations
            annotations['bboxes'] = annotations['bboxes'].copy()
            n_bboxes = annotations['bboxes'].shape[0]

            # Non parallelized option
            for index in range(n_bboxes):
                annotations['bboxes'][index, :] = deformable_transform_aabb(image.shape, deformable_transform, annotations['bboxes'][index, :])

            ## Parallelized option
            ## See: https://www.machinelearningplus.com/python/parallel-processing-python/

            ## Init multiprocessing.Pool()
            #pool = mp.Pool(mp.cpu_count())
            #results = [pool.apply_async(deformable_transform_aabb,
                                        #args=(image.shape,
                                              #deformable_transform,
                                              #annotations['bboxes'][i, :])
                                        #) for i in range(n_bboxes)]
            ## Close Pool and let all the processes complete
            #pool.close()
            #pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

            ## <results> is a list of pool.ApplyResult objects
            ## NOTE: This attribution is not ORDERED since async
            ##       is being used to parallelize the function
            #for i in range(n_bboxes):
                #annotations['bboxes'][i, :] = results[i].get()

            #-------------------------------------------
            # # Draw the annotations
            # draw_annotations(image,
            #                  annotations,
            #                  color=(255, 0, 0))
            # cv2.imshow("Middle", image)
            #-------------------------------------------

        # Randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            # Affine + flip transformation
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # Apply transformation to the image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        #-------------------------------------------
        # # Draw the annotations
        # from utils.visualization import draw_annotations
        # draw_annotations(image,
        #                  annotations,
        #                  color=(0, 255, 0))

        # # Show output image after data augmentation
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
        #-------------------------------------------

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """!@brief
        Randomly transforms each image and its annotations.
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """!@brief
        Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """!@brief
        Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """!@brief
        Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """!@brief
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """!@brief
        Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """!@brief
        Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def compute_input_output(self, group):
        """!@brief
        Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # Check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # Randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # Perform preprocessing steps:
        #  - Cast to float32
        #  - Subtract the mean values
        #  - Resize the image
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        #-------------------------------------------
        # # Draw the annotations
        # from utils.visualization import draw_annotations
        #
        # for image_index, image in enumerate(image_group):
        #     image = 255 * (image - np.min(image)) / np.ptp(image)
        #     image = image.astype(np.uint8)
        #     draw_annotations(image,
        #                     annotations_group[image_index],
        #                     color=(0, 255, 0))
        #
        #     # Show output image after data augmentation
        #     cv2.imshow("Preprocessed Image", image)
        #     cv2.waitKey(0)
        #-------------------------------------------

        # Compute network inputs
        inputs = self.compute_inputs(image_group)

        # Compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """!@brief
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """!@brief
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets
