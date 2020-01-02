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

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations, draw_box, draw_cross, draw_caption

import keras
import numpy as np
import os
from PIL import Image

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def _compute_ap(recall, precision, save_path):
    """!@brief
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    @param recall    : The recall curve (list).
    @param precision : The precision curve (list).
    @param save_path : Path to save the Precision-Recall Curve.

    @return
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    #--------------------------------------------------
    # Compute F1-score for each detection time (row)
    # OBS: This method is called for each label, so
    #      in the case where there is more than one
    #      class, the following plot must be adjusted.
    f1s = 2 * precision * recall / np.maximum((precision + recall), np.finfo(np.float64).eps)

    # WRONG:
    # if f1s.size:
    #     f1 = np.max(f1s)
    # else:
    #     f1 = 0.0

    # F1-Score is compute here for only the last item in the list of precision-recall,
    # because these lists are cumulative lists (each row represents the precision-recall
    # value for that specific number of detections).
    if precision.size and recall.size:
        f1 = 2 * precision[-1] * recall[-1] / np.maximum((precision[-1] + recall[-1]), np.finfo(np.float64).eps)
    else:
        f1 = 0.0

    # Saving arrays into a csv file
    if save_path is not None:
        # ensure directory created first
        makedirs(save_path)
        csv_name = os.path.join(save_path, "Precision-Recall_F1.csv")

        f = open(csv_name, "w")
        f.write("{},{},{}\n".format("Precision", "Recall", "F1-Score"))
        for x in zip(precision, recall, f1s):
            f.write("{},{},{}\n".format(x[0], x[1], x[2]))
        f.close()

        # Saving Precision-Recall Curve
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        path = os.path.join(save_path, "Precision-Recall_Curve.png")

        print("Saving precision-recall curve plot...")
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.title('Precision-Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(path, dpi=300, transparent=True)
        plt.close(fig)
    #---------------------------------

    return f1, ap


def _non_max_suppression(detections, nms_thres=0.5):
    """!@brief
    Performs Non-Maximum Suppression to further filter detections.

    @param detections : Model detections before NMS.
    @param nms_thres  : Threshold applied to the IoU between detections.

    @return
        array(x1,y1,x2,y2,score,class_pred)

    @see
        https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522
        https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py
    """
	# if there are no boxes, return an empty list
    if len(detections) == 0:
        return np.array([])

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if detections.dtype.kind == "i":
        detections = detections.astype("float")

    # # Debug:
    # print("")
    # for j in range(len(detections)):
    #     x1 = np.trunc(detections[j][0]).astype(int)
    #     y1 = np.trunc(detections[j][1]).astype(int)
    #     x2 = np.trunc(detections[j][2]).astype(int)
    #     y2 = np.trunc(detections[j][3]).astype(int)
    #     cs = detections[j][4]
    #     lb = np.trunc(detections[j][5]).astype(int)
    #     print("{}-Box: [{},{}]-[{},{}]\tConf-Score: {:.2f}\tLabel: {}".format(j,x1,y1,x2,y2,cs,lb))

    keep_boxes = []
    while len(detections):
        # @NOTE:
        #       compute_overlap = iou
        #       In the code:
        #           (iw * ih) = area of overlap
        #                  ua = union area
        large_overlap = compute_overlap(np.expand_dims(detections[0, :4], axis=0), detections[:, :4]) > nms_thres
        label_match = detections[0, -1] == detections[:, -1]

        # Indices of boxes with large IoUs and matching labels
        invalid = large_overlap & label_match
        weights = detections[invalid[0], 4:5]

        # Merge overlapping bboxes by order of confidence
        detections[0, :4] = (weights * detections[invalid[0], :4]).sum(0) / (weights.sum() + np.finfo(np.float).eps)
        keep_boxes += [detections[0]]
        detections = detections[~invalid[0]]

    return np.array(keep_boxes)


def _get_indexes_inside_mask(mask_base_path, image_name, boxes):
    """!@brief
    Compute the indexes inside the mask image when provided.

    @param mask_base_path : The base path to find the mask image (mask_name equals to image_name).
    @param image_name     : The image file name in the current batch generator.
    @param boxes          : Array of all anchor boxes [x1,y1,x2,y2] (detections or annotations).

    @return
        List of boxes indexes whose centroid is inside the mask image.
    """
    if mask_base_path is None or image_name is None:
        return None, np.arange(len(boxes))

    # search for the image mask to be used
    mask_path = None
    indexes = []
    files = os.listdir(mask_base_path)

    for f in files:
        file = f.split(".")[0]
        ext  = f.split(".")[1]
        if image_name.lower() == file.lower() and ext in ['png','jpg','jpeg','JPG']:
            mask_path = os.path.join(mask_base_path, f)

    if mask_path is None:
        print(" Mask image not found for image {}. Using results for the entire image.".format(image_name))
        return None, np.arange(len(boxes))
    else:
        # load mask image as grayscale
        mask = np.asarray(Image.open(mask_path).convert('L'))

        # select indexes that are inside mask region
        for idx, [x1, y1, x2, y2] in enumerate(boxes):
            # get boxes centroids
            x = np.round(x1 + ((x2-x1)/2)).astype(int)
            y = np.round(y1 + ((y2-y1)/2)).astype(int)
            if mask[y][x] > 0:
                indexes.append(idx)

        return mask, indexes


def _get_detections(
    generator,
    model,
    score_threshold=0.05,
    max_detections=100,
    nms_thres=0.1,
    save_path=None,
    mask_base_path=None
):
    """!@brief
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    @param generator       : The generator used to run images through the model.
    @param model           : The model to run on the images.
    @param score_threshold : The score confidence threshold to use.
    @param max_detections  : The maximum number of detections to use per image.
    @param nms_thres       : The threshold value applied to the IoU between detections.
    @param save_path       : The path to save the images with visualized detections to.
    @param mask_base_path  : The base path to find mask images accordingly.

    @return
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image_name   = generator.get_image_name(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indexes which have a score above the threshold
        indexes = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indexes]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indexes[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indexes[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # # Debug:
        # image_detections = np.array([
        # 	(12,  84, 140, 212, 0.99, 0),
        # 	(24,  84, 152, 212, 0.89, 0),
        # 	(36,  84, 164, 212, 0.79, 0),
        # 	(12,  96, 140, 224, 0.69, 0),
        # 	(24,  96, 152, 224, 0.59, 0),
        # 	(24, 108, 152, 236, 0.49, 0),
        #     (250,250, 290, 290, 0.90, 0)])
        # draw_detections(raw_image, image_detections[:,:4], image_detections[:,4], np.trunc(image_detections[:,-1]).astype(int), label_to_name=generator.label_to_name)
        # cv2.imshow("Image", raw_image)
        # cv2.waitKey()

        # print(image_detections[:, :-1])
        # print("-----------------------------------------------")

        # Apply NMS according to the box overlappings
        # ----------------------------------------
        image_detections = _non_max_suppression(image_detections, nms_thres=nms_thres)

        # # Debug:
        # if len(image_detections):
        #     print("After NMS:\n---------------------------------")
        #     for j in range(len(image_detections)):
        #         x1 = np.trunc(image_detections[j][0]).astype(int)
        #         y1 = np.trunc(image_detections[j][1]).astype(int)
        #         x2 = np.trunc(image_detections[j][2]).astype(int)
        #         y2 = np.trunc(image_detections[j][3]).astype(int)
        #         cs = image_detections[j][4]
        #         lb = np.trunc(image_detections[j][5]).astype(int)
        #         print("{}-Box: [{},{}]-[{},{}]\tConf-Score: {:.2f}\tLabel: {}".format(j,x1,y1,x2,y2,cs,lb))
        #     draw_detections(raw_image, image_detections[:,:4], image_detections[:,4], np.trunc(image_detections[:,-1]).astype(int), label_to_name=generator.label_to_name)
        #     cv2.imshow("NMS", raw_image)
        #     cv2.waitKey()
        #     exit()

        # Apply mask ROI to the detections
        # ----------------------------------------
        if mask_base_path is not None and len(image_detections) > 0:
            mask, inside_indexes = _get_indexes_inside_mask(mask_base_path, image_name, image_detections[:,:4])
            image_detections = image_detections[inside_indexes]

        # # Debug:
        # print(image_detections)
        # draw_detections(mask, image_detections[:,:4], image_detections[:,4], np.trunc(image_detections[:,-1]).astype(int), label_to_name=generator.label_to_name)
        # cv2.imshow("Remove boxes", mask)
        # cv2.waitKey()
        # exit(0)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            if len(image_detections):
                all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                all_detections[i][label] = np.array([])

    return all_detections


def _get_annotations(generator, mask_base_path=None):
    """!@brief
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    @param generator      : The generator used to retrieve ground truth annotations.
    @param mask_base_path : The base path to find mask images accordingly.

    @return
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # Apply mask ROI to the detections
        # ----------------------------------------
        if mask_base_path is not None:
            mask, inside_indexes = _get_indexes_inside_mask(mask_base_path, generator.get_image_name(i), annotations['bboxes'])
            annotations['bboxes'] = annotations['bboxes'][inside_indexes]
            annotations['labels'] = annotations['labels'][inside_indexes]

        # # Debug:
        # print(len(annotations['bboxes']), len(annotations['labels']))
        # draw_annotations(mask, annotations, color=(0,0,0), label_to_name=generator.label_to_name)
        # cv2.imshow("Remove boxes", mask)
        # cv2.waitKey()
        # exit(0)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=500,
    nms_thres=0.1,
    save_path=None,
    mask_base_path=None
):
    """!@brief
    Evaluate a given dataset using a given model.

    @param generator       : The generator that represents the dataset to evaluate.
    @param model           : The model to evaluate.
    @param iou_threshold   : The threshold used to consider when a detection is positive or negative.
    @param score_threshold : The score confidence threshold to use for detections.
    @param max_detections  : The maximum number of detections to use per image.
    @param nms_thres       : The threshold value applied to the IoU between detections.
    @param save_path       : The path to save images with visualized detections to.
    @param mask_base_path  : The base path to find mask images accordingly.

    @return
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator,
                                     model,
                                     score_threshold=score_threshold,
                                     max_detections=max_detections,
                                     nms_thres=nms_thres,
                                     save_path=save_path,
                                     mask_base_path=mask_base_path)
    all_annotations = _get_annotations(generator, mask_base_path)
    average_precisions = {}

    # count = 0
    # for i in range(len(all_detections)):
    #     count += len(all_detections[i][0])
    # print("Total of detections in all images: ", count)

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    # loop: each label in the model
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        # for each image in the batch with label 'label'
        for i in range(generator.size()):
            raw_image            = generator.load_image(i)
            detections           = all_detections[i][label] # d[0:3]=bbox, d[4]=score
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            # Draw manual annotations in the image, the np.c_ method
            # add a column (labels column) to the array of annotations
            draw_annotations(raw_image, np.c_[annotations, np.full(len(annotations), label)], color=(255,0,0), label_to_name=generator.label_to_name)

            if len(detections):
                # for each detection in the image 'i'
                for d in detections:
                    box = d[:4].astype(int)
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)   # is it correct to use 'compute_overlap'?
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)

                        # Draw a green box for TP
                        draw_box(raw_image, box, color=(0,255,0), thickness=1)
                        caption = generator.label_to_name(label) + ': {0:.2f}'.format(d[4])
                        #draw_caption(raw_image, box, caption)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

                        # Draw a red box for FP
                        draw_box(raw_image, box, color=(0,0,255), thickness=1)
                        caption = generator.label_to_name(label) + ': {0:.2f}'.format(d[4])
                        #draw_caption(raw_image, box, caption)

            # Save the image
            if save_path is not None:
                # ensure directory created first
                makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, '{}_label-{}.png'.format(i, label)), raw_image)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0, 0
            continue

        # sort by score
        indexes         = np.argsort(-scores)
        false_positives = false_positives[indexes]
        true_positives  = true_positives[indexes]

        # compute false positives and true positives
        #   Obs: cummulative sum -> np.cumsum([1,2,3,4,5]) = [1,3,6,10,15]
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        F1, average_precision  = _compute_ap(recall, precision, save_path)
        average_precisions[label] = recall[-1], precision[-1], F1, average_precision, num_annotations

    return average_precisions
