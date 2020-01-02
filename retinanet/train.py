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
"""

import argparse
import os
import sys
import warnings
import numpy as np
from matplotlib import pyplot as plt

import keras
import keras.preprocessing.image
from keras.utils.vis_utils import plot_model
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# # Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#     import mynet.bin  # noqa: F401
#     __package__ = "mynet.bin"

import layers  # noqa: F401
import models
from loss_function                import losses
from callbacks                    import RedirectModel
from callbacks.eval               import Evaluate, TrainValTensorBoard
from models.retinanet             import retinanet_bbox
from data_generator.csv_generator import CSVGenerator
from data_generator.kitti         import KittiGenerator
from data_generator.open_images   import OpenImagesGenerator
from data_generator.pascal_voc    import PascalVocGenerator
from data_generator.ivm           import IVMGenerator
from utils.anchors                import make_shapes_callback
from utils.config                 import read_config_file, parse_anchor_parameters
from utils.keras_version          import check_keras_version
from utils.model                  import freeze as freeze_model
from utils.transform              import random_transform_generator

from callbacks.cyclical_learning_rate import CyclicLR
from callbacks.learning_rate_finder import LearningRateFinder


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """!@brief
    Construct a modified tf session which attempts to
    allocate only as much GPU memory based on runtime allocations.
    """
    # Silence Tensorflow
    #   Levels:
    #       0: Everything
    #       1: Warnings
    #       2: Errors
    #       3: Fatal erros
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """!@brief
    Load weights for model.

    @param model         : The model to load weights for.
    @param weights       : The weights to load.
    @param skip_mismatch : If True, skips layers whose shape of weights doesn't
                           match with the model.

    @return
        The model with loaded weights.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet,
                  num_classes,
                  weights,
                  multi_gpu=2,
                  freeze_backbone=False,
                  lr=1e-5,
                  config=None):
    """!@brief
    Creates three models (model, training_model, prediction_model).

    @param backbone_retinanet : A function to call to create a retinanet model with a given backbone.
    @param num_classes        : The number of classes to train.
    @param weights            : The weights to load into the model.
    @param multi_gpu          : The number of GPUs to use for training.
    @param freeze_backbone    : If True, disables learning for the backbone.
    @param config             : Config parameters, None indicates the default configuration.

    @returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection
                           (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight
    # sharing, and to prevent OOM errors.
    # Optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/gpu:1'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
                                       weights=weights,
                                       skip_mismatch=True)

        training_model = multi_gpu_model(model, gpus=multi_gpu)  # , cpu_merge=False) # <- recommended for NV-Link
    else:
        model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
                                   weights=weights,
                                   skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001),
        metrics=['accuracy']
    )

    return model, training_model, prediction_model


def create_callbacks(model,
                     training_model,
                     prediction_model,
                     validation_generator,
                     train_size,
                     lr_range,
                     args):
    """!@brief
    Creates the callbacks to use during training.

    @param model                : The base model.
    @param training_model       : The model that is used for training.
    @param prediction_model     : The model that should be used for validation.
    @param validation_generator : The generator for creating validation data.
    @param train_size           : Number of trainable images.
    @param lr_range             : Learning rate range for Cyclical-LR.
    @param args                 : parseargs args object.

    @return
        A list of callbacks used for training.
    """
    callbacks = []

    # Set callback for Tensorboard
    #-----------------------------
    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = TrainValTensorBoard(  # keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    # Set callback for evaluation on validation dataset
    #-----------------------------
    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator,
                                  iou_threshold    = args.iou_threshold,
                                  score_threshold  = args.score_threshold,
                                  max_detections   = args.max_detections,
                                  tensorboard      = tensorboard_callback,
                                  weighted_average = args.weighted_average,
                                  save_path        = args.eval_img_path)

        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # Set callback for saving the model
    #-----------------------------
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                #'{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
                '{backbone}_{dataset_type}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
            #monitor='val_loss',
            #verbose=1,
            #save_best_only=True,
            #save_weights_only=False,
            #mode='auto',
            #period=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    # Set callback for the learning rate schedule
    #-----------------------------
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss', # train_loss
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    # Initialize the cyclical learning rate callback
    #   Modes: 'triangular', 'triangular2', 'exp_range'
    #-----------------------------
    step_size = 8
    callbacks.append(CyclicLR(
    	mode='exp_range',
    	base_lr=lr_range[0],
    	max_lr=lr_range[1],
    	step_size= step_size * (train_size // args.batch_size)
    ))

    # # NOTE: Don't use with Cyclical Learning Rate callback
    # # Stop training when a monitored quantity has stopped improving
    # #-----------------------------
    # callbacks.append(keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=10,
    #     verbose=1,
    #     mode='auto',
    #     baseline=None,
    #     restore_best_weights=True
    # ))

    # Set callback for save training information in a csv file
    #-----------------------------
    if args.csv_log_path:
        # ensure directory created first
        makedirs(args.csv_log_path)
        callbacks.append(keras.callbacks.CSVLogger(
            os.path.join(args.csv_log_path, 'training_log.csv'),
            append=True
        ))

    return callbacks


def create_generators(args, preprocess_image):
    """!@brief
    Create generators for training and validation.

    @param args             : Parseargs object containing configuration for generators.
    @param preprocess_image : Function that preprocesses an image for the network.

    @return
        The generators created.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
        'photometric'      : args.random_photometric,
        'motion'           : args.random_motion,
        'deformable'       : args.random_deformable,
        'alpha'            : (1,200),
        'sigma'            : (4,7),
    }

    # create random transform generator for augmenting training data
    # returns a matrix of transformation ramdonly generated (yield)
    transform_generator = None
    if args.random_transform:
        transform_generator = random_transform_generator(
            # min_rotation=-0.1,
            # max_rotation=0.1,
            # min_translation=(-0.1, -0.1),
            # max_translation=(0.1, 0.1),
            # min_shear=-0.1,
            # max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator = transform_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            **common_args
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator = transform_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            **common_args
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator = transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            args.main_dir,
            subset               = 'train',
            version              = args.version,
            labels_filter        = args.labels_filter,
            annotation_cache_dir = args.annotation_cache_dir,
            parent_label         = args.parent_label,
            transform_generator  = transform_generator,
            **common_args
        )

        validation_generator = OpenImagesGenerator(
            args.main_dir,
            subset               = 'validation',
            version              = args.version,
            labels_filter        = args.labels_filter,
            annotation_cache_dir = args.annotation_cache_dir,
            parent_label         = args.parent_label,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        train_generator = KittiGenerator(
            args.kitti_path,
            subset='train',
            transform_generator = transform_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            args.kitti_path,
            subset='val',
            **common_args
        )
    elif args.dataset_type == 'ivm':
        train_generator = IVMGenerator(
            args.ivm_path,
            'train',
            transform_generator = transform_generator,
            **common_args
        )

        validation_generator = IVMGenerator(
            args.ivm_path,
            'val',
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """!@brief
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    @param parsed_args : parser.parse_args()

    @return
            parsed_args.
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """!@brief
    Parse the arguments.
    """
    #--------------------
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    ivm_parser = subparsers.add_parser('ivm')
    ivm_parser.add_argument('ivm_path', help='Path to IVM dataset directory (ie. /data/dataset).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir',               help='Path to dataset directory.')
    oid_parser.add_argument('--version',              help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',        help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label',         help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations',       help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes',           help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    #--------------------

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',            help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',    help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',             help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',          help='Don\'t initialize the model with any weights.', action='store_const', const=False)

    parser.add_argument('--backbone',           help='Backbone model used by retinanet.', default='resnet101', type=str)
    parser.add_argument('--batch-size',         help='Size of the batches.', default=2, type=int)
    parser.add_argument('--gpu',                help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',          help='Number of GPUs to use for parallel processing.', type=int, default=2)
    parser.add_argument('--multi-gpu-force',    help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true', default='True')
    parser.add_argument('--epochs',             help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--steps',              help='Number of steps per epoch.', type=int, default=290)
    parser.add_argument('--lr',                 help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--find-lr',            help='Learning rate automatic finder option', action='store_false', default='False')
    parser.add_argument('--snapshot-path',      help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--csv-log-path',       help='Path to store a log file with all model measures during training.', default=None)
    parser.add_argument('--tensorboard-dir',    help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',       help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',      help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--eval-img-path',      help='The path to save images of evaluation with visualized detections to.', default=None)
    parser.add_argument('--freeze-backbone',    help='Freeze training of backbone layers.', action='store_true', default='True')
    parser.add_argument('--random-photometric', help='Randomly apply photometric distortions to the image.', action='store_true')
    parser.add_argument('--random-motion',      help='Randomly apply motion blur to the image.', action='store_true')
    parser.add_argument('--random-transform',   help='Randomly apply affine + flip transformations to the image and annotations.', action='store_true')
    parser.add_argument('--random-deformable',  help='Randomly apply smooth elastic deformation to the image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',     help='Rescale the image so the smallest side is min_side.', type=int, default=1200)
    parser.add_argument('--image-max-side',     help='Rescale the image if the largest side is larger than max_side.', type=int, default=2000)
    parser.add_argument('--config',             help='Path to configuration parameters .ini file.', default=None)
    parser.add_argument('--weighted-average',   help='Compute the mAP using the weighted average of precisions among classes.', action='store_true', default='True')

    # Fit generator arguments
    parser.add_argument('--workers',        help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=16)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args, 'training')
    #     print("----------------------------------")
    #     print("ARGUMENTS IN CONFIG FILE:")
    #     for sec in args.config.sections():
    #         print(sec, "=", dict(args.config.items(sec)))
    #     print("----------------------------------")
    #
    # for arg in vars(args):
    #     print(arg, "=", getattr(args, arg))
    # exit()

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # # Debuging
    # for i in range(1):
    #     inputs, targets = train_generator.__getitem__(i)
    # exit()

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)

        # When using as a second step for fine-tuning
        for layer in model.layers:
            layer.trainable = True

        training_model = model
        anchor_params  = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

        ###################################################################################### BRUNO


        # compile model
        training_model.compile(
            loss={
                'regression'    : losses.smooth_l1(),
                'classification': losses.focal()
            },
            optimizer=keras.optimizers.adam(lr=args.lr, clipnorm=0.001)
        )
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config
        )

    # Print model design
    # print(model.summary())
    # print(training_model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # exit()

    # Get the number of samples in the training and validations datasets.
    train_size = train_generator.size()
    val_size   = validation_generator.size()
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, val_size, args.batch_size))

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        train_size,
        [1e-6, 1e-4],
        args,
    )

    # Use multiprocessing if workers > 0
    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    # check to see if we are attempting to find an optimal learning rate
    # before training for the full number of epochs
    if args.find_lr:
        # initialize the learning rate finder and then train with learning
        # rates ranging from 1e-10 to 1e+1
        print("[INFO] Finding learning rate...")
        lrf = LearningRateFinder(training_model)
        lrf.find(train_generator, 1e-10, 1e+1,
            stepsPerEpoch=np.ceil((train_size / float(args.batch_size))),
            batchSize=args.batch_size)

        # plot the loss for the various learning rates and save the
        # resulting plot to disk
        lrf.plot_loss()
        plt.savefig("lrfind_plot.png")

        # save values into a csv file
        lrf.save_csv("lr_loss.csv")

        # gracefully exit the script so we can adjust our learning rates
        # in the config and then train the network for our full set of
        # epochs
        print("[INFO] Learning rate finder complete")
        print("[INFO] Examine plot and adjust learning rates before training")
        sys.exit(0)

    # Number of epochs and steps for training new layers
    n_epochs = 350
    steps = train_size // args.batch_size

    # start training
    training_model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size
    )

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        # for layer in model.layers:
        #     if layer.name is 'bn_conv1':
        #         print("Before\t-> Trainable: {}, Freeze: {}".format(layer.trainable, layer.freeze))

        for layer in model.layers:
            layer.trainable = True

        # recompile to apply the change
        model.compile(
            loss={
                'regression'    : losses.smooth_l1(),
                'classification': losses.focal()
            },
            # Learning rate must be lower for training the entire network
            optimizer=keras.optimizers.adam(lr=args.lr*0.1, clipnorm=0.001),
            metrics=['accuracy']
        )

        if args.multi_gpu > 1:
            from keras.utils import multi_gpu_model
            with tf.device('/gpu:1'):
                training_model = multi_gpu_model(model, gpus=args.multi_gpu)
        else:
            training_model = model

        # recompile to apply the change
        training_model.compile(
            loss={
                'regression'    : losses.smooth_l1(),
                'classification': losses.focal()
            },
            # Learning rate must be lower for training the entire network
            optimizer=keras.optimizers.adam(lr=args.lr*0.1, clipnorm=0.001),
            metrics=['accuracy']
        )
        print('Unfreezing all layers.')

        # for layer in model.layers:
        #     if layer.name is 'bn_conv1':
        #         print("After\t-> Trainable: {}, Freeze: {}".format(layer.trainable, layer.freeze))

        # Print training_model design
        # print(model.summary())
        # print(training_model.summary())

        # create the callbacks
        callbacks = create_callbacks(
            model,
            training_model,
            prediction_model,
            validation_generator,
            train_size,
            [1e-8, 1e-6],
            args,
        )

        batch_size = 2 # note that more GPU memory is required after unfreezing the body
        steps = train_size // batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, val_size, batch_size))
        training_model.fit_generator(
            generator=train_generator,
            validation_data=validation_generator,
            steps_per_epoch=steps,
            epochs=args.epochs,
            initial_epoch=n_epochs,
            verbose=1,
            callbacks=callbacks,
            workers=args.workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=args.max_queue_size
        )

if __name__ == '__main__':
    main()
