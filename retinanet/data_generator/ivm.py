"""!
Copyright (C) 2019 Bruno Gregorio - BIPG

    https://brunoggregorio.github.io \n
    https://www.bipgroup.dc.ufscar.br

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

from .generator import Generator
from utils.image import read_image_bgr

import os
import numpy as np
from six import raise_from
from PIL import Image

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

ivm_classes = {'cell':0}


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class IVMGenerator(Generator):
    """!@brief
    Generate data for the IVM dataset.
    """

    def __init__(
        self,
        data_dir,
        set_name,
        classes=ivm_classes,
        image_extension='.jpg',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        """!@brief
        Initialize the IVM data generator.

        @param base_dir : Directory w.r.t. where the files are to be searched
                         (defaults to the directory containing the video directories).
        """
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'image_sets', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(IVMGenerator, self).__init__(**kwargs)

    def size(self):
        """!@brief
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """!@brief
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """!@brief
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """!@brief
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """!@brief
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """!@brief
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """!@brief
        Compute the aspect ratio for an image with image_index.
        """
        path  = os.path.join(self.data_dir, 'frames', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """!@brief
        Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'frames', self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path)

    def get_image_name(self, image_index):
        """!@brief
        Get image file name at the image_index.
        """
        return self.image_names[image_index]

    def __parse_annotation(self, element):
        """!@brief
        Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox    = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float)
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float)
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float)
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float)

        return truncated, difficult, box, label

    def __parse_annotations(self, xml_root):
        """!@brief
        Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((len(xml_root.findall('object')),)), 'bboxes': np.empty((len(xml_root.findall('object')), 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'][i, :] = box
            annotations['labels'][i] = label

        return annotations

    def load_annotations(self, image_index):
        """!@brief
        Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
