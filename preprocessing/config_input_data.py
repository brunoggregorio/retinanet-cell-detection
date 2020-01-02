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

# TODO:
#   Fix number of files with 4 decimal (video1_0001.jpg)

import os
import gc
import cv2
import argparse
import random
import numpy as np
import xml.etree.ElementTree as xml

# Read the input parameters
ap = argparse.ArgumentParser(description="Create a structured folder containing the data from a video file.")

ap.add_argument("-i", "--video",         required=True,  help="Input video used to create the output structured folder")
ap.add_argument("-g", "--ground_truth",  required=True,  help="Input manual annotated centroids")
ap.add_argument("-o", "--output_folder", required=True,  help="Output root folder name")
ap.add_argument("-r", "--radius",        required=False, help="Cell radius value (approximately)", default=10)

ap.add_argument("--voc",  help="Create an output similar to the VOC Pascal dataset", action="store_true")
ap.add_argument("--coco", help="Create an output similar to the MS COCO dataset",    action="store_true")
ap.add_argument("--yolo", help="Create an output to be used in YOLOv3 algorithm",    action="store_true")

args = ap.parse_args()

##********************************************************
##  Checking paths
##********************************************************

# Check video path
if os.path.exists(args.video):
    video_path = os.path.abspath(args.video)
else:
    print("Cannot find " + args.video)

# Check ground truth path
if os.path.exists(args.ground_truth):
    gt_path = os.path.abspath(args.ground_truth)
else:
    print("Cannot find " + args.ground_truth)

# Check output path
if os.path.exists(args.output_folder):
    out_path = os.path.abspath(args.output_folder)
else:
    print("Creating output folder...")
    try:
        os.mkdir(args.output_folder)
        out_path = os.path.abspath(args.output_folder)
    except OSError:
        print ("Creation of the directory %s failed" % args.output_folder)

#print(video_path)
#print(gt_path)
#print(out_path)

#********************************************************
#   Create output folder tree
#********************************************************

# Enter into the folder
os.chdir(out_path)
print("Creating folders in %s/" % out_path)

# Check subfolders
if os.path.isdir('frames') == False :
    try:
        os.mkdir('frames')
    except OSError:
        print ("Creation of the directory 'frames' failed")

if os.path.isdir('annotations') == False :
    try:
        os.mkdir('annotations')
    except OSError:
        print ("Creation of the directory 'annotations' failed")

if os.path.isdir('image_sets') == False :
    try:
        os.mkdir('image_sets')
    except OSError:
        print ("Creation of the directory 'image_sets' failed")


#********************************************************
#   Save video frames into frames folder
#********************************************************

# Enter into the frames folder
frames_path = out_path + '/frames'
os.chdir(frames_path)

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Read until video is completed
f = 1
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        img_name = os.path.split(os.path.splitext(args.video)[0])[1] + "__{}.jpg".format(f)
        cv2.imwrite(img_name, frame)
        f += 1

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()


#********************************************************
#   Create and save xml annotations file
#********************************************************
# Enter into the frames folder
os.chdir(out_path + '/annotations')

# Read ground truth values
ann = np.loadtxt(gt_path, dtype='i')
f = ann[:, 0]
xcen = ann[:, 1]
ycen = ann[:, 2]

i = 0
f_tmp = 0

r = int(args.radius)

# Vector to store bboxes of the first frame
# just for visual tests
first_bboxes = []

while i < f.size:
    # Check if the object is in the same frame
    if f[i] != f_tmp:  # New frame
        # If not the first case write the file
        if i > 0:
            tree = xml.ElementTree(root)
            with open(xml_name, "w") as fh:
                tree.write(xml_name, encoding='utf-8')

        # Get frame name
        img_name = frames_path + "/" + os.path.split(os.path.splitext(args.video)[0])[1] + "__" + str(f[i]) + ".jpg"
        frame = cv2.imread(img_name)
        h, w, d = frame.shape # returns the number of rows, columns and channels

        # Create a new XML file with annotations
        xml_name = os.path.split(os.path.splitext(args.video)[0])[1] + "__" + str(f[i]) + ".xml"

        root = xml.Element("annotation")

        folder = xml.SubElement(root, "folder")
        folder.text = os.path.split(os.path.splitext(args.video)[0])[1]

        filename = xml.SubElement(root, "filename")
        filename.text = os.path.split(os.path.splitext(args.video)[0])[1] + "__" + str(f[i]) + ".jpg"

        # ---------- Source ------------
        source = xml.Element("source")
        root.append(source)

        database = xml.SubElement(source, "database")
        database.text = "IVM"

        annotation = xml.SubElement(source, "annotation")
        annotation.text = os.path.split(os.path.splitext(args.video)[0])[1]

        image = xml.SubElement(source, "image")
        image.text = "IVM"

        flickrid = xml.SubElement(source, "flickrid")   # Remove this
        flickrid.text = "?"
        # -----------------------------

        # ---------- Owner ------------                 # Irrelevant section
        owner = xml.Element("owner")
        root.append(owner)

        flickrid = xml.SubElement(owner, "flickrid")
        flickrid.text = "?"

        name = xml.SubElement(owner, "name")
        name.text = "BIPG"
        # ----------------------------

        # ---------- Size ------------
        size = xml.Element("size")
        root.append(size)

        width = xml.SubElement(size, "width")
        width.text = str(w)

        height = xml.SubElement(size, "height")
        height.text = str(h)

        depth = xml.SubElement(size, "depth")
        depth.text = str(d)
        # ---------------------------

        # -------- Objects ----------
        obj = xml.Element("object")
        root.append(obj)

        name = xml.SubElement(obj, "name")
        name.text = "cell"

        pose = xml.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = xml.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = xml.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = xml.Element("bndbox")
        obj.append(bndbox)

        # Fixed window size
        # TODO:
        #   Choose a better way to set the size window according to
        #   the size of the cell (scale) to properly train the network
        xmin = xml.SubElement(bndbox, "xmin")
        xmin.text = str( np.clip(xcen[i]-r, 0, w-1) )

        ymin = xml.SubElement(bndbox, "ymin")
        ymin.text = str( np.clip(ycen[i]-r, 0, h-1) )

        xmax = xml.SubElement(bndbox, "xmax")
        xmax.text = str( np.clip(xcen[i]+r, 0, w-1) )

        ymax = xml.SubElement(bndbox, "ymax")
        ymax.text = str( np.clip(ycen[i]+r, 0, h-1) )

        # Draw the resulting bbox in the image frame
        #cv2.rectangle(frame,
                    #(xcen[i]-r, ycen[i]-r),
                    #(xcen[i]+r, ycen[i]+r),
                    #(0, 255, 0),
                    #1)
        #cv2.imshow("Frame", frame)
        #cv2.waitKey(0)
        # ---------------------------

        # Store the resulting bbox for the first frame printing
        if f[i] == 1:
            x_coord = (xcen[i]-r, ycen[i]-r)
            y_coord = (xcen[i]+r, ycen[i]+r)
            coords = tuple([x_coord, y_coord])
            first_bboxes.append(coords)

        f_tmp = f[i]
    else:   # Same frame
        # Continue into the same XML file adding new objects

        # -------- Objects ----------
        obj = xml.Element("object")
        root.append(obj)

        name = xml.SubElement(obj, "name")
        name.text = "cell"

        pose = xml.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = xml.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = xml.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = xml.Element("bndbox")
        obj.append(bndbox)

        # Fixed window size
        # TODO:
        #   Choose a better way to set the size window according to
        #   the size of the cell (scale) to properly train the network
        xmin = xml.SubElement(bndbox, "xmin")
        xmin.text = str( np.clip(xcen[i]-r, 0, w-1) )

        ymin = xml.SubElement(bndbox, "ymin")
        ymin.text = str( np.clip(ycen[i]-r, 0, h-1) )

        xmax = xml.SubElement(bndbox, "xmax")
        xmax.text = str( np.clip(xcen[i]+r, 0, w-1) )

        ymax = xml.SubElement(bndbox, "ymax")
        ymax.text = str( np.clip(ycen[i]+r, 0, h-1) )

        # Draw the resulting bbox in the image frame\
        #cv2.rectangle(frame,
                    #(xcen[i]-r, ycen[i]-r),
                    #(xcen[i]+r, ycen[i]+r),
                    #(0, 255, 0),
                    #1)
        #cv2.imshow("Frame", frame)
        #cv2.waitKey(0)
        # ---------------------------

        # Store the resulting bbox for the first frame printing
        if f[i] == 1:
            x_coord = (xcen[i]-r, ycen[i]-r)
            y_coord = (xcen[i]+r, ycen[i]+r)
            coords = tuple([x_coord, y_coord])
            first_bboxes.append(coords)

    i += 1

# Save the last xml file
tree = xml.ElementTree(root)
with open(xml_name, "w") as fh:
    tree.write(xml_name, encoding='utf-8')

cv2.destroyAllWindows()


# --------------------------
# Draw the first frame just for visual analysis

# Enter the right path to save the image
os.chdir(out_path)

# Get frame name
img_name = frames_path + "/" + os.path.split(os.path.splitext(args.video)[0])[1] + "__1.jpg"
frame = cv2.imread(img_name)

# Iterate between the rectangles
for x, y in first_bboxes:
    cv2.rectangle(frame, x, y, (0, 255, 0), 1)

img_name = "Frame_1_example_r" + str(args.radius) + ".png"
cv2.imwrite(img_name, frame)
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
cv2.imshow(img_name, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------


#********************************************************
#   Separate the data into train, val and test sets
#********************************************************

# Total samples -> 80% for trainval and 20% for test
n_samples = max(f)
n_trainval = round(n_samples * 0.8)
n_test = n_samples - n_trainval
n_train = round(n_trainval * 0.8)
n_val = n_trainval - n_train

print("----------------------------------------------------------------")
print("Total number of samples: {:.0f}".format(n_samples))
print("\t20% for test set: {:.0f} samples".format(n_test))
print("\t80% for trainval set: {:.0f} samples".format(n_trainval))
print("\t\t80% of trainval for train: {:.0f} samples".format(n_train))
print("\t\t20% of trainval for val: {:.0f} samples".format(n_val))
print("----------------------------------------------------------------")

# Create an array with the image names
img_list = []
for img_file_name in os.listdir(frames_path):
    if img_file_name.endswith(".png") or img_file_name.endswith(".jpg") or img_file_name.endswith(".jpeg"):
        img_list.append(os.path.splitext(img_file_name)[0])

# Randomize to create random sets of data
random.shuffle(img_list)

# Function to sample and remove items from a list
def list_sample(data, n):
    output = []
    for i in range(0, n):
        index = random.randrange(len(data))
        output.append(data.pop(index))
    return output

# Test set
test_set = list_sample(img_list, int(n_test))
#print("* Test set:\n\t", test_set)

# Trainval set
trainval_set = list_sample(img_list, int(n_trainval))
#print("\n* Trainval set:\n\t", trainval_set)

## Train set
trainval_aux = trainval_set.copy()
train_set = list_sample(trainval_aux, int(n_train))
#print("\n* Train set:\n\t", train_set)

## Val set
val_set = list_sample(trainval_aux, int(n_val))
#print("\n* Val set:\n\t", val_set)

# Enter into the frames folder
img_sets_path = out_path + '/image_sets'
os.chdir(img_sets_path)

# Save lists into text files
np.savetxt('test.txt',     test_set,     fmt='%s')
np.savetxt('trainval.txt', trainval_set, fmt='%s')
np.savetxt('train.txt',    train_set,    fmt='%s')
np.savetxt('val.txt',      val_set,      fmt='%s')
