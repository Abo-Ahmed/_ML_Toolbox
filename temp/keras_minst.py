  
def cnn_minst_model():
  # a- loading the data set
  print('>>> loading dataset...')
  mnist = tf.keras.datasets.mnist

  # b- data normalization
  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # c- model building with simple sequential CNN
  print('>>> building NN model...')

  # model = tf.keras.models.Sequential([
  #   tf.keras.layers.Flatten(input_shape=(28, 28)),
  #   tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #   tf.keras.layers.Dropout(0.2),
  #   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  # ])

  # anther way for creating model 
  model = tf.keras.Sequential()
  # Adds a densely-connected layer with 64 units to the model:
  model.add(layers.Flatten(input_shape=(28, 28)))
  model.add(layers.Dense(512, activation=tf.nn.relu))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(10, activation=tf.nn.softmax))

  # d- model configuration
  print('>>> configuring NN model...')
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  # e- model training over the dataset
  print('>>> training NN model...')
  model.fit(x_train, y_train, epochs=2)

  # f- model testing results
  print('>>> testing NN model...')
  print (model.evaluate(x_test, y_test)) 
  print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

def use_gpu () :
  with tf.device('/gpu:0'):  
    print('>>> GPU is running')
    cnn_minst_model()
    
def use_cpu () :
  with tf.device('/cpu:0'): 
    print('>>> CPU is running')
    cnn_minst_model()


# # start of main program code 
# gpu_time = timeit.timeit('use_gpu()', number=1, setup="from __main__ import use_gpu")
# cpu_time = timeit.timeit('use_cpu()', number=1, setup="from __main__ import use_cpu")

# print('>>> time required for this task on GPU...')
# print(gpu_time)
# print('>>> time required for this task on CPU...')
# print(cpu_time)
# print('GPU speedup over CPU: {}x'.format(round(cpu_time/gpu_time , 1 )))
