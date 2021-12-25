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
from google.colab import drive , files
from sklearn.metrics import confusion_matrix
from __future__ import print_function
from PIL import Image
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.client import device_lib
from tensorflow.keras import Model, Input , activations
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