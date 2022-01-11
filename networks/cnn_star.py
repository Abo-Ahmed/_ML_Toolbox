import PIL
import os
import random
import cv2
import timeit
import time
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as keras_layers
from tensorflow.python.client import device_lib
from keras.applications.vgg16 import VGG16
from keras.layers.wrappers import TimeDistributed
from google.colab import drive , files
from sklearn.metrics import confusion_matrix
from __future__ import print_function
from psutil import virtual_memory

def getGitRepo(name):
  !rm -rf '/content/'$name
  !git clone 'https://github.com/Abo-Ahmed/'$name

getGitRepo('_master')
execfile('/content/_master/main.py')