import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import timeit
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from __future__ import print_function
from urllib.request import urlretrieve

def download(url, file):
    if not os.path.isfile(file):
        print(">>> Download file... " + file + " ...")
        urlretrieve(url,file)
        print(">>> File downloaded")
    else:
        print ('>>> h5 model is already downloaded!')

# Root directory of the project
print('>>> downloading network from github')
!pip install -q xlrd
!git clone https://github.com/GantMan/nsfw_model

print('>>> downloading h5 model')
download('https://s3.amazonaws.com/nsfwdetector/nsfw.299x299.h5','nsfw_detector.h5')
print('>>> setting the local directory')
ROOT_DIR = os.path.abspath("/content/nsfw_model")
sys.path.append(ROOT_DIR)  # To find local version of the library
from nsfw_detector import NSFWDetector

# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()

def cnn_NSFW_model(imgs):
    print ('>>> start NSFW detector')
    detector = NSFWDetector('/content/nsfw_detector.h5')
    # Predict multiple images at once using Keras batch prediction
    print ('>>> prediction of group')
    results = detector.predict(imgs, batch_size=2)
    return (results)

!pip install -q xlrd
!git clone https://github.com/matterport/Mask_RCNN
# Root directory of the project
ROOT_DIR = os.path.abspath("/content/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model_m = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model_m.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def show_results(result):
  print('>>> showing resutls')
  print(result)
  for x in result:
    # display_one(image, result[x])
    image = skimage.io.imread(x)
    # Run detection
    results = model_m.detect([image], verbose=1)
    r = results[0]
    print(result[x])
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    
cnn_result = cnn_NSFW_model(['/content/11.jpg', '/content/12.jpg','/content/13.jpg', '/content/14.jpg','/content/15.jpg', '/content/16.jpg'])
show_results(cnn_result)
