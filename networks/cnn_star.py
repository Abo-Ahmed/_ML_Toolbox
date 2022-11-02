
# dependencies
# !pip install tensorflow==2.4.1
# !pip install keras==2.4.0
# !pip install -q git+https://github.com/tensorflow/docs


from __future__ import print_function
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
from keras.regularizers import l2
from google.colab import drive , files
from sklearn.metrics import confusion_matrix

def getGitRepo(name):
  !rm -rf '/content/'$name
  !git clone 'https://github.com/Abo-Ahmed/'$name
  execfile('/content/_master_network/main.py')

getGitRepo('_master_network')
