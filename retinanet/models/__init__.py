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
"""

from __future__ import print_function
import sys


class Backbone(object):
    """!@brief
    This class stores additional information on backbones.
    """
    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        import layers
        from loss_function import losses
        from initialization import initializers
        self.custom_objects = {
            'UpsampleLike'     : layers.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : layers.RegressBoxes,
            'FilterDetections' : layers.FilterDetections,
            'Anchors'          : layers.Anchors,
            'ClipBoxes'        : layers.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_focal'           : losses.focal(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """!@brief
        Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """!@brief
        Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """!@brief
        Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """!@brief
        Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    """!@brief
    Returns a backbone object for the given backbone.
    """
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone_name:
        from .mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    """!@brief
    Loads a retinanet model using the correct custom objects.

    @param filepath : one of the following:
                        - string, path to the saved model, or
                        - h5py.File object from which to load the model
    @param backbone_name : Backbone with which the model was trained.

    @return
        A keras.models.Model object.

    @throws
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models
    return keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)


def convert_model(model, nms=True, class_specific_filter=True, anchor_params=None):
    """!@brief
    Converts a training model to an inference model.

    @param model                 : A retinanet training model.
    @param nms                   : Boolean, whether to add NMS filtering to the converted model.
    @param class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
    @param anchor_params         : Anchor parameters object. If omitted, default values are used.

    @return
        A keras.models.Model object.

    @throws
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    from .retinanet import retinanet_bbox
    return retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter, anchor_params=anchor_params)


def assert_training_model(model):
    """!@brief
    Assert that the model is a training model.
    """
    assert(all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
    """!@brief
    Check that model is a training model and exit otherwise.
    """
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
