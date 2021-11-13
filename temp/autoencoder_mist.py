#auto-encoder with minst with keras
# to enable GPU: Runtime > chnage runtime accelerator > gpu
# Load libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import timeit
from keras.layers import Input,Dense
from keras.models import Model,Sequential
from keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from __future__ import print_function

def show_version () :
  print('>>> versions Info:')
  print('>>> Tenserflow version: ' + tf.__version__)
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('>>> Tenserflow Device Name: ' + device_name )
  print('>>> Keras version: ' + tf.keras.__version__)
  print('>>> List of all local devices:')
  local_device_protos = device_lib.list_local_devices()
  print( [ x.name for x in local_device_protos ])
  print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

def autoencoder_minst_model():
  # a- loading the data set
  print('>>> loading dataset...')
  (X_train,_), (X_test,_)=mnist.load_data()

  # b- data normalization
  X_train=X_train.astype('float32')/float(X_train.max())
  X_test=X_test.astype('float32')/float(X_test.max())
  X_train=X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
  X_test=X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
  print("Training set : ",X_train.shape) #The resolution has changed
  print("Testing set : ",X_test.shape)

  # c- model building with simple sequential CNN
  print('>>> building Auto-encoder model...')
  input_dim=X_train.shape[1]
  encoding_dim=32
  compression_factor=float(input_dim/encoding_dim)
  autoencoder=Sequential()
  autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,),activation='relu'))
  autoencoder.add(Dense(input_dim,activation='sigmoid'))
  input_img=Input(shape=(input_dim,))
  encoder_layer=autoencoder.layers[0]
  encoder=Model(input_img,encoder_layer(input_img))
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(X_train,X_train,epochs=50, batch_size=256, shuffle=True, validation_data=(X_test,X_test))

  # d- test image and prediction
  print('>>> Testnig Auto-encoder model...')
  num_images=10
  np.random.seed(42)
  random_test_images=np.random.randint(X_test.shape[0], size=num_images)
  encoded_img=encoder.predict(X_test)
  decoded_img=autoencoder.predict(X_test)

  # e- visualize the model prediction
  plt.figure(figsize=(18,4))
  for i, image_idx in enumerate(random_test_images):
      #plot input image
      ax=plt.subplot(3,num_images,i+1)
      plt.imshow(X_test[image_idx].reshape(28,28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      # plot encoded image
      ax = plt.subplot(3, num_images, num_images + i + 1)
      plt.imshow(encoded_img[image_idx].reshape(8, 4))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      # plot reconstructed image
      ax = plt.subplot(3, num_images, 2*num_images + i + 1)
      plt.imshow(decoded_img[image_idx].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()

def use_gpu () :
  with tf.device('/gpu:0'):  
    print('>>> GPU is running')
    autoencoder_minst_model()
    
# start of main program code 
show_version()   
gpu_time = timeit.timeit('use_gpu()', number=1, setup="from __main__ import use_gpu")
print('>>> time required for this task on GPU...')
print(gpu_time)
