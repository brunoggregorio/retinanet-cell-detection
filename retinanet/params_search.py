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

# import argparse
import os
import sys

import keras
import tensorflow as tf

import models
from data_generator.ivm import IVMGenerator
from utils.eval import evaluate
from utils.keras_version import check_keras_version

import sklearn
import numpy as np
import pandas as pd
import shutil

def get_session():
    """!@brief
    Construct a modified tf session.
    """
    # Silence Tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def evaluate_model(generator=None,
                   model=None,
                   nt=0.5,
                   it=0.25,
                   st=0.4,
                   md=500,
                   save_path=None,
                   mask=None,
                   output=[]):
    # evaluate the model
    average_precisions = evaluate(
        generator=generator,
        model=model,
        nms_thres       = nt,  # 0.4 IoU threshold for NMS between detections (candidates)
        iou_threshold   = it,  # 0.2 [AP_25=0.25, AP_50=0.50, AP_75=0.75, AP=avg(0.50,0.05,0.95)]
        score_threshold = st,  # 0.45 confidence value
        max_detections  = md,  # fixed value (max=331 from 'eae-cerebro-0014_t001_10003.jpg')
        save_path       = save_path, # '/home/bgregorio/workspace/mynet_keras/out_imgs/test',
        mask_base_path  = mask # '/home/bgregorio/workspace/data/dataset/all/masks'
    )

    # print evaluation
    for l, (r, p, f1, ap, num_annotations) in average_precisions.items():
        dic = {'nms_threshold': nt,
               'iou_threshold': it,
               'score_threshold': st,
               'max_detections': md,
               'label': l,
               'recall': r,
               'precision': p,
               'f1_score': f1,
               'average_precision': ap,
               'num_annotations': num_annotations}
        output.append(dic)


def main(args=None):

    iou_thres_vec = []
    if sys.argv[1:] == []:
        print('Please give at least one value for IoU threshold.')
        exit(0)
    for v in sys.argv[1:]:
        iou_thres_vec.append(float(v))
    print('Doing job for IoU values:', iou_thres_vec)

    # make sure keras is the minimum required version
    check_keras_version()

    # define base paths
    model_base_path = '/path/snapshots/'
    data_base_path = '/path/dataset/'
    out_imgs_base_path = '/path/out_imgs/'

    # define number of experiments and k-folds
    experiments = ['1']#,'2','3','4']
    kfolds = ['1']#,'2','3','4','5']

    # loop for each experiment and k-fold
    for exp in experiments:
        # set experiment name
        if   exp == '1': data_name = 'cns_stratified'
        elif exp == '2': data_name = 'cns_unseen_split'
        elif exp == '3': data_name = 'all_stratified'
        elif exp == '4': data_name = 'all_unseen_split'

        for kf in kfolds:
            # set names
            model_name = 'resnet101_fpn4_1000_sc4_ar3_cycLRexp-8_allDataAugm_350-450_exp' + exp + '_k' + kf
            model_path = model_base_path + model_name + '/resnet101_ivm.h5'
            data_path = data_base_path + 'image_sets/' + data_name + '/k_' + kf
            save_path = out_imgs_base_path + model_name

            # copy data image_set to data_base_path
            files = os.listdir(data_path)
            for f in files:
                if f.endswith('.txt'):
                    f = data_path + '/' + f
                    dst = data_base_path + 'image_sets/'
                    # print('File copied\n\tFROM:', f, '\n\tTO:', dst)
                    shutil.copy(f, dst)

            # optionally choose specific GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            keras.backend.tensorflow_backend.set_session(get_session())

            # load the model
            print('Loading model, this may take a second...')
            model = models.load_model(model_path, backbone_name='resnet101')

            # convert the model
            model = models.convert_model(model)
            print('Loaded.')

            # create the generator
            generator = IVMGenerator(
                data_base_path,
                'val',
                image_min_side=1000,
                image_max_side=1400
            )

            #
            # Grid search
            #
            md = 500
            nms_thres_vec = np.arange(.05, 1., .05)
            score_thres_vec = np.arange(.05, 1., .05)

            # output values
            outputs = []

            # varying params
            for it in iou_thres_vec:
                for nt in nms_thres_vec:
                    for st in score_thres_vec:
                        evaluate_model(generator=generator,
                                       model=model,
                                       nt=nt,
                                       it=it,
                                       st=st,
                                       md=md,
                                       save_path=None,
                                       mask='/path/dataset/all/masks',
                                       output=outputs)

            # save in data frame format
            df = pd.DataFrame(data=outputs)

            # ensure directory created first and save file
            makedirs(save_path)
            out_path = save_path + '/params_search.csv'
            df.to_csv(out_path, index=None, header=True)

            #
            # Get best set of parameters
            #
            # compute measure
            avg_measure = (df['f1_score'] + df['average_precision']) / 2
            df['avg_measure'] = avg_measure

            # set best param values
            best = df.iloc[df['avg_measure'].idxmax()]
            md = best['max_detections'].astype(int)
            it = best['iou_threshold']
            nt = best['nms_threshold']
            st = best['score_threshold']

            # create the generator for test dataset
            generator = IVMGenerator(
                data_base_path,
                'test',
                image_min_side=1000,
                image_max_side=1400
            )

            # perform inference in test dataset
            outputs = []

            # varying params
            evaluate_model(generator=generator,
                           model=model,
                           nt=nt,
                           it=it,
                           st=st,
                           md=md,
                           save_path=save_path,
                           mask='/path/dataset/all/masks',
                           output=outputs)

            # save in data frame format
            df = pd.DataFrame(data=outputs)
            out_path = save_path + '/best_output.csv'
            df.to_csv(out_path, index=None, header=True)

if __name__ == '__main__':
    main()
