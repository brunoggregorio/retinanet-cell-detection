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

import cv2
import numpy as np

from .colors import label_color

def draw_cross(image, box, color=(0, 255, 0), thickness=1, size=3):
    """!@brief
    Draws a cross on an image with a given color.

    @param image     : The image to draw on.
    @param box       : A list of 4 elements (x1, y1, x2, y2).
    @param color     : The color of the cross.
    @param thickness : The thickness of the lines to draw a cross with.
    """
    b = np.array(box)
    cross = ( ((b[0]+b[2])/2).astype(int), ((b[1]+b[3])/2).astype(int) )
    cv2.drawMarker(image, cross,  color, cv2.MARKER_CROSS, size, thickness)


def draw_box(image, box, color, thickness=1):
    """!@brief
    Draws a box on an image with a given color.

    @param image     : The image to draw on.
    @param box       : A list of 4 elements (x1, y1, x2, y2).
    @param color     : The color of the box.
    @param thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """!@brief
    Draws a caption above the box in an image.

    @param image   : The image to draw on.
    @param box     : A list of 4 elements (x1, y1, x2, y2).
    @param caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=1):
    """!@brief
    Draws boxes on an image with a given color.

    @param image     : The image to draw on.
    @param boxes     : A [N, 4] matrix (x1, y1, x2, y2).
    @param color     : The color of the boxes.
    @param thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.0):
    """!@brief
    Draws detections in an image.

    @param image           : The image to draw on.
    @param boxes           : A [N, 4] matrix (x1, y1, x2, y2).
    @param scores          : A list of N classification scores.
    @param labels          : A list of N labels.
    @param color           : The color of the boxes. By default the color from
                             keras_retinanet.utils.colors.label_color will be used.
    @param label_to_name   : (optional) Functor for mapping a label to a name.
    @param score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """!@brief
    Draws annotations in an image.

    @param image         : The image to draw on.
    @param annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary
                           containing bboxes (shaped [N, 4]) and labels (shaped [N]).
    @param color         : The color of the boxes. By default the color from
                           keras_retinanet.utils.colors.label_color will be used.
    @param label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        #draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
