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
import numpy as np
from utils.eval import evaluate

class Evaluate(keras.callbacks.Callback):
    """!@brief
    Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """!@brief
        Evaluate a given dataset using a given model at the end of every epoch during training.

        @param generator        : The generator that represents the dataset to evaluate.
        @param iou_threshold    : The threshold used to consider when a detection is positive or negative.
        @param score_threshold  : The score confidence threshold to use for detections.
        @param max_detections   : The maximum number of detections to use per image.
        @param save_path        : The path to save images with visualized detections to.
        @param tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
        @param weighted_average : Compute the mAP using the weighted average of precisions among classes.
        @param verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator        = generator
        self.iou_threshold    = iou_threshold
        self.score_threshold  = score_threshold
        self.max_detections   = max_detections
        self.save_path        = save_path
        self.tensorboard      = tensorboard
        self.weighted_average = weighted_average
        self.verbose          = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        F1s = []
        for label, (F1, average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision),
                      'and F1-score: {:.4f}'.format(F1))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            F1s.append(F1)

        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        self.mean_f1 = sum(F1s) / sum(x > 0 for x in total_instances)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf

            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = np.float_(self.mean_ap)
            summary_value.tag = "mAP"

            summary_value = summary.value.add()
            summary_value.simple_value = np.float_(self.mean_f1)
            summary_value.tag = "mF1"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = np.float_(self.mean_ap)
        logs['mF1'] = np.float_(self.mean_f1)

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
            print('mF1: {:.4f}'.format(self.mean_f1))



################################################################################
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context

class TrainValTensorBoard(TensorBoard):
    """!
    @note https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure?rq=1
    """
    def __init__(self, log_dir='./logs', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
