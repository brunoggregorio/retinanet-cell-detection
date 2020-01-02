#!/usr/bin/env python

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

Example of usage:
    python debug.py --anchors --annotations --image-min-side=1000 --image-max-side=1400 ivm /path/dataset/
"""

import argparse
import os
import sys
import cv2

# # Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#     import mynet.bin  # noqa: F401
#     __package__ = "mynet.bin"

# Change these to absolute imports if you copy this script outside the package.
from data_generator.pascal_voc    import PascalVocGenerator
from data_generator.csv_generator import CSVGenerator
from data_generator.kitti         import KittiGenerator
from data_generator.open_images   import OpenImagesGenerator
from data_generator.ivm           import IVMGenerator
from utils.keras_version          import check_keras_version
from utils.transform              import random_transform_generator
from utils.visualization          import draw_annotations, draw_boxes
from utils.anchors                import anchors_for_shape, compute_gt_annotations
from utils.config                 import read_config_file, parse_anchor_parameters


def create_generator(args):
    """!@brief
    Create the data generators.

    @param args: parseargs arguments object.
    """
    common_args = {
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'photometric'      : args.random_photometric,
        'motion'           : args.random_motion,
        'deformable'       : args.random_deformable,
        'alpha'            : (1,200),
        'sigma'            : (4,7),
    }

    transform_generator = None

    # create random transform generator for augmenting training data
    # returns a matrix of transformation ramdonly generated (yield)
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.pascal_set,
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'oid':
        generator = OpenImagesGenerator(
            args.main_dir,
            subset=args.subset,
            version=args.version,
            labels_filter=args.labels_filter,
            parent_label=args.parent_label,
            annotation_cache_dir=args.annotation_cache_dir,
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        generator = KittiGenerator(
            args.kitti_path,
            subset=args.subset,
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'ivm':
        generator = IVMGenerator(
            args.ivm_path,
            args.ivm_set,
            transform_generator=transform_generator,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    """!@brief
    Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--pascal-set',  help='Name of the set to show (defaults to test).', default='test')

    ivm_parser = subparsers.add_parser('ivm')
    ivm_parser.add_argument('ivm_path', help='Path to dataset directory (ie. /data/dataset/all).')
    ivm_parser.add_argument('--ivm-set',  help='Name of the set to show (defaults to val).', default='val')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
    kitti_parser.add_argument('subset', help='Argument for loading a subset from train/val.')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('subset', help='Argument for loading a subset from train/validation/test.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('-l', '--loop',         help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--no-resize',          help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors',            help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations',        help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-photometric', help='Randomly apply photometric distortions to the image.', action='store_true')
    parser.add_argument('--random-motion',      help='Randomly apply motion blur to the image.', action='store_true')
    parser.add_argument('--random-transform',   help='Randomly apply affine + flip transformations to the image and annotations.', action='store_true')
    parser.add_argument('--random-deformable',  help='Randomly apply smooth elastic deformation to the image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',     help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',     help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',             help='Path to a configuration parameters .ini file.')

    return parser.parse_args(args)


def run(generator, args, anchor_params):
    """!@brief
    Main loop.

    @param generator : The generator to debug.
    @param args      : Parseargs args object.
    """
    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)

        # Apply random transformations
        # if args.random_transform or args.random_deformable or args.random_photometric:# or args.random_psf_blur:
        if True:
            image, annotations = generator.random_transform_group_entry(image, annotations)

        # resize the image and annotations
        if args.resize:
            image, image_scale = generator.resize_image(image)
            annotations['bboxes'] *= image_scale

        anchors = anchors_for_shape(image.shape, anchor_params=anchor_params)
        positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

        # draw anchors on the image
        if args.anchors:
            draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

        # draw annotations on the image
        if args.annotations:
            # draw annotations in red
            draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

            # draw regressed anchors in green to override most red annotations
            # result is that annotations without anchors are red, with anchors are green
            draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            return False
    return True


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args, 'debug')

    # make sure keras is the minimum required version
    check_keras_version()

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # create the display window
    #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    if args.loop:
        while run(generator, args, anchor_params=anchor_params):
            pass
    else:
        run(generator, args, anchor_params=anchor_params)


if __name__ == '__main__':
    main()
