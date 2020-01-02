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
    python evaluate.py --model=/path/resnet101_ivm.h5 --convert-model --image-min-side=1000 --image-max-side=1400 ivm /path/dataset/
"""

import argparse
import os
import sys

import keras
import tensorflow as tf

# # Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#     import mynet.bin  # noqa: F401
#     __package__ = "mynet.bin"

import models
from data_generator.csv_generator import CSVGenerator
from data_generator.pascal_voc import PascalVocGenerator
from data_generator.ivm import IVMGenerator
from utils.config import read_config_file, parse_anchor_parameters
from utils.eval import evaluate
from utils.keras_version import check_keras_version

# retirar
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.image import read_image_bgr, preprocess_image, resize_image


def get_session():
    """ Construct a modified tf session.
    """
    # Silence Tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'ivm':
        validation_generator = IVMGenerator(
            args.ivm_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    ivm_parser = subparsers.add_parser('ivm')
    ivm_parser.add_argument('ivm_path', help='Path to IVM dataset directory (ie. /data/dataset).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('--model',            help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true', default='True')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet101')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', type=int, default=0)
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.03).', default=0.03, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.2).', default=0.2, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=1200)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=2000)
    parser.add_argument('--mask-folder',      help='Folder path for the mask images when available.', default=None)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args, 'evaluation')
        #print("----------------------------------")
        #print("ARGUMENTS IN CONFIG FILE:")
        #for sec in args.config.sections():
            #print(sec, "=", dict(args.config.items(sec)))
        #print("----------------------------------")

    # for arg in vars(args):
    #     print(arg, "=", getattr(args, arg))
    # exit()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    # print model summary
    # print(model.summary())

    # layer_outputs = []
    # layer_names = ['res2c_relu',                # C2
    #                'res3b3_relu',               # C3
    #                'res4b22_relu',              # C4
    #                'P2',                        # P2
    #                'P3',                        # P3
    #                'P4',                        # P4
    #                # 'regression_submodel',       # Subreg
    #                # 'classification_submodel',   # SubClas
    #                'regression',                # Regression
    #                'classification']            # Classification
    #
    # for layer in model.layers:
    #     if layer.name in layer_names:
    #         print('------------------------------------------------------------------------------------------------------------------')
    #         print('Layer found: ', layer.name)
    #         print('\tOutput:', layer.output)
    #         print('------------------------------------------------------------------------------------------------------------------')
    #         layer_outputs.append(layer.output)
    #
    # image = preprocess_image(generator.load_image(0))
    # image, scale = resize_image(image, args.image_min_side, args.image_max_side)
    #
    # activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    # activations = activation_model.predict(np.expand_dims(image, axis=0))
    #
    # def display_activation(activations, col_size, row_size, act_index):
    #     activation = activations[act_index]
    #     activation_index=0
    #     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    #     for row in range(0,row_size):
    #         for col in range(0,col_size):
    #             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
    #             activation_index += 1
    #             plt.savefig('layer_{}.png'.format(layer_names[act_index]))
    #
    # display_activation(activations, 8, 8, 0)
    # display_activation(activations, 8, 8, 1)
    # display_activation(activations, 8, 8, 2)
    # display_activation(activations, 8, 8, 3)
    # display_activation(activations, 8, 8, 4)
    # display_activation(activations, 8, 8, 5)
    #
    # exit()

    # start evaluation
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        average_precisions = evaluate(
            generator,
            model,
            iou_threshold   = args.iou_threshold,
            score_threshold = args.score_threshold,
            max_detections  = args.max_detections,
            save_path       = args.save_path,
            mask_base_path  = args.mask_folder
        )

        # print evaluation
        total_instances = []
        precisions = []
        F1s = []
        for label, (recall, precision, F1, average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label),
                  'with average precision: {:.4f}'.format(average_precision),
                  'precision: {:.4f}'.format(precision),
                  'recall: {:.4f}'.format(recall),
                  'and F1-score: {:.4f}'.format(F1))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            F1s.append(F1)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))
        print('mF1: {:.4f}'.format(sum(F1s) / sum(x > 0 for x in total_instances)))


if __name__ == '__main__':
    main()
