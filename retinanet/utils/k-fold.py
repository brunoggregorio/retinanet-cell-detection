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


import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


def save_k_kold_image_sets(k=5,
                           sets=['trainval','train','val','test'],
                           folders=None,
                           output_path=None,
                           shuffle=True):
    """!@brief
    Save the image set files according to the cross-validation requested and
    compute K-Fold algorithm.

    @param k           : Number of folds.
    @param sets        : List with the dataset names to be used.
    @param folders     : Folders containing the image files.
    @param output_path : Path to the image set files output.
    @param shuffle     : Whether to shuffle each class’s samples before splitting into batches.

    @return
        All the image set files accordingly to the K-fold algorithm requested.
    """
    # Asserts
    for f in folders:
        assert os.path.isdir(f)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read image files in folders
    images = []
    classes = []
    for folder in folders:
        # r=root, d=directories, f=files
        for r,d,f in os.walk(folder):
            for file in f:
                if ('.jpg' or '.png') in file.lower():
                    images.append(os.path.splitext(file)[0])
                    classes.append(folder)

    #
    x = np.array(images)
    y = np.array(classes)
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle)

    # Create train, validation and test datasets
    iter=1
    if 'val' in sets and 'test' in sets:
        for trainval_idx, test_idx in skf.split(x,y):
            # Trainval and test datasets
            x_trainval, x_test = x[trainval_idx], x[test_idx]
            y_trainval, y_test = y[trainval_idx], y[test_idx]

            # Train and validation datasets (set as 80/20 of trainval)
            # NOTE: Passing only in one split.
            skf_tmp = StratifiedKFold(n_splits=k, shuffle=True)
            for train_idx, val_idx in skf_tmp.split(x_trainval, y_trainval):
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                continue

            # print("TRAINVAL:", x[trainval_idx], "\nTRAIN:", x_trainval[train_idx], "\nVAL:", x_trainval[val_idx], "\nTEST:", x[test_idx])

            print("K-Fold (k={}) iteration {}:".format(k, iter))
            print("Total number of input images: {}".format(len(x)))
            print("#trainval: {}, #train: {}, #val: {}, #test: {}".format(len(x_trainval), len(x_train), len(x_val), len(x_test)))

            # Create output folders
            out_folder = 'k_' + str(iter)
            k_out_path = os.path.join(output_path, out_folder)
            if not os.path.exists(k_out_path):
                os.mkdir(k_out_path)

            # Saving image set files
            np.savetxt(k_out_path+'/trainval.txt', x[trainval_idx], fmt="%s")
            np.savetxt(k_out_path+'/train.txt', x_trainval[train_idx], fmt="%s")
            np.savetxt(k_out_path+'/val.txt', x_trainval[val_idx], fmt="%s")
            np.savetxt(k_out_path+'/test.txt', x[test_idx], fmt="%s")

            iter += 1
    elif 'test' in sets:
        for train_idx, test_idx in skf.split(x,y):
            # Train and test datasets
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # print("TRAIN:", x[train_idx], "\nTEST:", x[test_idx])

            print("K-Fold (k={}) iteration {}:".format(k, iter))
            print("Total number of input images: {}".format(len(x)))
            print("#train: {}, #test: {}".format(len(x_train), len(x_test)))

            # Create output folders
            out_folder = 'k_' + str(iter)
            k_out_path = os.path.join(output_path, out_folder)
            if not os.path.exists(k_out_path):
                os.mkdir(k_out_path)

            # Saving image set files
            np.savetxt(k_out_path+'/train.txt', x[train_idx], fmt="%s")
            np.savetxt(k_out_path+'/test.txt', x[test_idx], fmt="%s")

            iter += 1
    elif 'test' not in sets:
        for train_idx, val_idx in skf.split(x,y):
            # Trainval and test datasets
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # print("TRAIN:", x[train_idx], "\nVAL:", x[val_idx])

            print("K-Fold (k={}) iteration {}:".format(k, iter))
            print("Total number of input images: {}".format(len(x)))
            print("#train: {}, #val: {}".format(len(x_train), len(x_val)))

            # Create output folders
            out_folder = 'k_' + str(iter)
            k_out_path = os.path.join(output_path, out_folder)
            if not os.path.exists(k_out_path):
                os.mkdir(k_out_path)

            # Saving image set files
            np.savetxt(k_out_path+'/trainval.txt', x, fmt="%s")
            np.savetxt(k_out_path+'/train.txt', x[train_idx], fmt="%s")
            np.savetxt(k_out_path+'/val.txt', x[val_idx], fmt="%s")

            iter += 1

folders = [
    'path_to_video_1/frames',
    'path_to_video_2/frames',
    'path_to_video_3/frames'
    ]

output_path = '/output_path'
save_k_kold_image_sets(k=5, folders=folders, output_path=output_path, sets=['trainval','train','val','test'])
