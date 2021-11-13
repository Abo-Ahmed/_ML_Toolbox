import PIL
import os
import random
import cv2
import timeit
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from google.colab import files
from sklearn.metrics import confusion_matrix
from __future__ import print_function
from PIL import Image
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.client import device_lib
from tensorflow.keras import Model, Input ,layers , activations
from tensorflow.keras.layers import Conv2D, Convolution2D , ConvLSTM2D 
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D ,MaxPooling3D
from tensorflow.keras.layers import Dense, ReLU, BatchNormalization , ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D , GlobalAveragePooling2D 
from tensorflow.keras.layers import Dropout, Flatten , Activation  , Add
from tensorflow.keras.layers import LSTM , Bidirectional , Reshape
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.datasets import mnist
from keras.applications.vgg16 import VGG16
from keras.layers.wrappers import TimeDistributed


def getGitRepo(name):
  !rm -rf '/content/'$name
  !git clone 'https://github.com/Abo-Ahmed/'$name

getGitRepo('_master')
execfile('/content/_master/main.py')