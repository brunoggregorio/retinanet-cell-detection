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

import configparser
import numpy as np
import keras
from .anchors import AnchorParameters

def to_bool(value):
    """!@brief
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() in ("yes", "y", "true",  "t", "1"): return True
    if str(value).lower() in ("no",  "n", "false", "f", "0", "0.0", "", "none", "[]", "{}"): return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))


def read_config_file(args, mode):
    config = configparser.ConfigParser()
    config.read(args.config)

    if str(mode).lower() == 'training':
        if 'data' in config:
            if 'random_photometric' in config['data']: args.random_photometric = to_bool(config['data']['random_photometric'])
            if 'random_motion'      in config['data']: args.random_motion      = to_bool(config['data']['random_motion'])
            if 'random_deformable'  in config['data']: args.random_deformable  = to_bool(config['data']['random_deformable'])
            if 'random_transform'   in config['data']: args.random_transform   = to_bool(config['data']['random_transform'])
            if 'image_min_side'     in config['data']: args.image_min_side     = int(config['data']['image_min_side'])
            if 'image_max_side'     in config['data']: args.image_max_side     = int(config['data']['image_max_side'])

        if 'weights' in config:
            if 'snapshot'         in config['weights']: args.snapshot         = config['weights']['snapshot']
            if 'imagenet_weights' in config['weights']: args.imagenet_weights = to_bool(config['weights']['imagenet_weights'])
            if 'weights'          in config['weights']: args.weights          = to_bool(config['weights']['weights'])
            if 'no_weights'       in config['weights']: args.no_weights       = to_bool(config['weights']['no_weights'])

        if 'model' in config:
            if 'backbone'        in config['model']: args.backbone        = config['model']['backbone']
            if 'batch_size'      in config['model']: args.batch_size      = int(config['model']['batch_size'])
            if 'epochs'          in config['model']: args.epochs          = int(config['model']['epochs'])
            if 'steps'           in config['model']: args.steps           = int(config['model']['steps'])
            if 'lr'              in config['model']: args.lr              = float(config['model']['lr'])
            if 'find_lr'         in config['model']: args.find_lr         = to_bool(config['model']['find_lr'])
            if 'freeze_backbone' in config['model']: args.freeze_backbone = to_bool(config['model']['freeze_backbone'])

        if 'callbacks' in config:
            if 'snapshots'        in config['callbacks']: args.snapshots        = to_bool(config['callbacks']['snapshots'])
            if 'snapshot_path'    in config['callbacks']: args.snapshot_path    = config['callbacks']['snapshot_path']
            if 'tensorboard_dir'  in config['callbacks']: args.tensorboard_dir  = config['callbacks']['tensorboard_dir']
            if 'csv_log_path'     in config['callbacks']: args.csv_log_path     = config['callbacks']['csv_log_path']
            if 'evaluation'       in config['callbacks']: args.evaluation       = to_bool(config['callbacks']['evaluation'])
            if args.evaluation:
                if 'eval_img_path'    in config['callbacks']: args.eval_img_path    = config['callbacks']['eval_img_path']
                if 'weighted_average' in config['callbacks']: args.weighted_average = to_bool(config['callbacks']['weighted_average'])
                if 'evaluation' in config:
                    if 'score_threshold' in config['evaluation']: args.score_threshold = float(config['evaluation']['score_threshold'])
                    if 'iou_threshold'   in config['evaluation']: args.iou_threshold   = float(config['evaluation']['iou_threshold'])
                    if 'max_detections'  in config['evaluation']: args.max_detections  = int(config['evaluation']['max_detections'])

        if 'processing' in config:
            if 'multi_gpu'       in config['processing']: args.multi_gpu       = int(config['processing']['multi_gpu'])
            if 'multi_gpu_force' in config['processing']: args.multi_gpu_force = to_bool(config['processing']['multi_gpu_force'])
            if 'workers'         in config['processing']: args.workers         = int(config['processing']['workers'])
            if 'max_queue_size'  in config['processing']: args.max_queue_size  = int(config['processing']['max_queue_size'])
    elif str(mode).lower() == 'evaluation':
        if 'model' in config:
            if 'backbone' in config['model']: args.backbone = config['model']['backbone']

        if 'data' in config:
            if 'image_min_side' in config['data']: args.image_min_side = int(config['data']['image_min_side'])
            if 'image_max_side' in config['data']: args.image_max_side = int(config['data']['image_max_side'])

        if 'evaluation' in config:
            if 'model'           in config['evaluation']: args.model           = config['evaluation']['model']
            if 'convert_model'   in config['evaluation']: args.convert_model   = to_bool(config['evaluation']['convert_model'])
            if 'gpu'             in config['evaluation']: args.gpu             = config['evaluation']['gpu']
            if 'score_threshold' in config['evaluation']: args.score_threshold = float(config['evaluation']['score_threshold'])
            if 'iou_threshold'   in config['evaluation']: args.iou_threshold   = float(config['evaluation']['iou_threshold'])
            if 'max_detections'  in config['evaluation']: args.max_detections  = int(config['evaluation']['max_detections'])
            if 'save_path'       in config['evaluation']: args.save_path       = config['evaluation']['save_path']
            if 'mask_folder'     in config['evaluation']: args.mask_folder     = config['evaluation']['mask_folder']
    elif str(mode).lower() == 'debug':
        if 'data' in config:
            if 'random_photometric' in config['data']: args.random_photometric = to_bool(config['data']['random_photometric'])
            if 'random_motion'      in config['data']: args.random_motion      = to_bool(config['data']['random_motion'])
            if 'random_deformable'  in config['data']: args.random_deformable  = to_bool(config['data']['random_deformable'])
            if 'random_transform'   in config['data']: args.random_transform   = to_bool(config['data']['random_transform'])
            if 'image_min_side'     in config['data']: args.image_min_side     = int(config['data']['image_min_side'])
            if 'image_max_side'     in config['data']: args.image_max_side     = int(config['data']['image_max_side'])
        if 'debug' in config:
            if 'loop'        in config['debug']: args.loop        = to_bool(config['debug']['loop'])
            if 'no_resize'   in config['debug']: args.no_resize   = to_bool(config['debug']['no_resize'])
            if 'anchors'     in config['debug']: args.anchors     = to_bool(config['debug']['anchors'])
            if 'annotations' in config['debug']: args.annotations = to_bool(config['debug']['annotations'])
    else:
        raise Exception('Invalid mode to read the config file ' + str(args.config))

    return config


def parse_anchor_parameters(config):
    """!@brief
    Parse the parameters that cannot be passed through arg parser in main
    """
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)
