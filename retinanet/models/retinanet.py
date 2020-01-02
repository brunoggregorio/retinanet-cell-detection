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

import keras
from initialization import initializers
import layers
from utils.anchors import AnchorParameters
from . import assert_training_model


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """!@brief
    Creates the default regression submodel.

    @param num_classes                 : Number of classes to predict a score for at each feature level.
    @param num_anchors                 : Number of anchors to predict classification scores for at each feature level.
    @param pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
    @param classification_feature_size : The number of filters to use in the layers in the classification submodel.
    @param name                        : The name of the submodel.

    @return
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """!@brief
    Creates the default regression submodel.

    @param num_values              : Number of values to regress.
    @param num_anchors             : Number of anchors to regress for each feature level.
    @param pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
    @param regression_feature_size : The number of filters to use in the layers in the regression submodel.
    @param name                    : The name of the submodel.

    @return
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C2, C3, C4, C5, feature_size=256):
    """!@brief
    Creates the FPN layers on top of the backbone features.

    @param C2           : Feature stage C2 from the backbone.
    @param C3           : Feature stage C3 from the backbone.
    @param C4           : Feature stage C4 from the backbone.
    @param C5           : Feature stage C5 from the backbone.
    @param feature_size : The feature size to use for the resulting feature levels.

    @return
        A list of feature levels [P2, P3, P4, P5].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3           = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, C2])
    P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # add P3 elementwise to C2
    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    # # "P6 is obtained via a 3x3 stride-2 conv on C5"
    # P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    # P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    # P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P2, P3, P4, P5]


def default_submodels(num_classes, num_anchors):
    """!@brief
    Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    @param num_classes : Number of classes to use.
    @param num_anchors : Number of base anchors.

    @return
        A list of tuple, where the first element is the name of the submodel and
        the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """!@brief
    Applies a single submodel to each FPN level.

    @param name     : Name of the submodel.
    @param model    : The submodel to evaluate.
    @param features : The FPN features.

    @return
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """!@brief
    Applies all submodels to each FPN level.

    @param models   : List of sumodels to run on each pyramid level (by default
                      only regression, classifcation).
    @param features : The FPN features.

    @return
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """!@brief
    Builds anchors for the shape of the features from FPN.

    @param anchor_parameters : Parameteres that determine how anchors are generated.
    @param features          : The FPN features.

    @returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors             = None,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """!@brief
    Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    @param inputs                  : keras.layers.Input (or list of) for the input to the model.
    @param num_classes             : Number of classes to classify.
    @param num_anchors             : Number of base anchors.
    @param create_pyramid_features : Functor for creating pyramid features given the features C2, C3, C4, C5 from the backbone.
    @param submodels               : Submodels to run on each feature map (default is regression and classification submodels).
    @param name                    : Name of the model.

    @returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C2, C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C2, C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    **kwargs
):
    """!@brief
    Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    @param model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
    @param nms                   : Whether to use non-maximum suppression for the filtering step.
    @param class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
    @param name                  : Name of the model.
    @param anchor_params         : Struct containing anchor parameters. If None, default values are used.
    @param *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    @returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P2', 'P3', 'P4', 'P5']]
    anchors  = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
