
def use_gpu () :
  with tf.device('/gpu:0'):  
    print('>>> GPU is running')
    cnn_training_model()
    
def use_cpu () :
  with tf.device('/cpu:0'): 
    print('>>> CPU is running')
    cnn_training_model()

def run_test_case():
#   gpu_time = timeit.timeit('use_gpu()', number=1, setup="from __main__ import use_gpu")
#   print('>>> time required for this task on GPU...\n' , gpu_time)
  cpu_time = timeit.timeit('use_cpu()', number=1, setup="from __main__ import use_cpu")
  print('>>> time required for this task on CPU...\n' , cpu_time)  
#   print('GPU speedup over CPU: {}x'.format(round(cpu_time/gpu_time , 2 )))

def read_images_folder(destnation_path):
  imageSet = []
  items = os.listdir(destnation_path)
  print (items)    
  for each_image in items:
    if each_image.endswith(".jpg"):
      imageSet.append( np.resize( cv2.cvtColor(cv2.imread(destnation_path + "/" + each_image),cv2.COLOR_BGR2GRAY).shape  , (150 , 300 )))
  return np.array(imageSet)

def read_special_image(destnation_path):
  return  np.resize( cv2.cvtColor(cv2.imread(destnation_path),cv2.COLOR_BGR2GRAY).shape  , (150 , 300 ))

def load_dataset_test() :
  print('>>> loading data - test ...')
  cats_path = '/content/drive/My Drive/Colab Notebooks/test_set/cats' 
  dogs_path = '/content/drive/My Drive/Colab Notebooks/test_set/dogs' 
  testCats = read_images_folder(cats_path);
  testDogs = read_images_folder(dogs_path);

  print('testCats: ', len(testCats))
  print('testDogs: ', len(testDogs))
  return ( testCats , testDogs )

def load_dataset_train() :
  print('>>> loading data - training ...')
  cats_path = '/content/drive/My Drive/Colab Notebooks/training_set/cats' 
  dogs_path = '/content/drive/My Drive/Colab Notebooks/training_set/dogs' 
  trainingCats = read_images_folder(cats_path);
  trainingDogs = read_images_folder(dogs_path);
  
  print('trainingCats: ', len(trainingCats))
  print('trainingDogs: ', len(trainingDogs))
  return  trainingCats , trainingDogs  

def build_model() :
  print('>>> building NN model...')

  # temp = tf.keras.models.Sequential([
  #   tf.keras.layers.Flatten(input_shape=(28, 28)),
  #   tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #   tf.keras.layers.Dropout(0.2),
  #   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  # ])

  # anther way for creating model 
  # Adds a densely-connected layer with 64 units to the model:
  temp = tf.keras.Sequential()
  temp.add(layers.Flatten(input_shape=(150, 300)))
  temp.add(layers.Dense(512, activation=tf.nn.relu))
  temp.add(layers.Dropout(0.2))
  temp.add(layers.Dense(10, activation=tf.nn.softmax))
  
  print('>>> configuring NN model...')
  temp.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  return temp

def cnn_training_model():
  
  # a- loading the data set , b- data normalization
  cat_test , dog_test  = load_dataset_test()
  
  x_test =  np.random.rand(2000,150,300)
  x_test[0:1000,:,:] = cat_test
  x_test[1000:2000,:,:] = dog_test
  x_test = x_test / 255.0
  
  y_test = np.random.rand(2000)
  y_test[0:1000] = np.ones(len(cat_test))
  y_test[1000:2000] = np.zeros(len(dog_test))
  
  cat_train , dog_train  = load_dataset_train()
  
  x_train =  np.random.rand(8000,150,300)
  x_train[0:4000,:,:] = cat_train
  x_train[4000:8000,:,:] = dog_train
  x_train = x_train / 255.0
  
  y_train = np.random.rand(8000)
  y_train[0:4000] = np.ones(len(cat_train))
  y_train[4000:8000] = np.zeros(len(dog_train))
  
  # c- model building and configuring with simple sequential CNN
  model = build_model()

  # e- model training over the dataset
  print('>>> training NN model...')
  model.fit(x_train, y_train, epochs=10)

  # f- model testing results
  print('>>> testing NN model...')
  print (model.evaluate(x_test, y_test))
  
  print('>>> special testing NN model...')
  s_x_test =  np.random.rand(1,150,300)
  s_x_test[0] = read_special_image('/content/drive/My Drive/Colab Notebooks/special_test/cat1.jpg')
  s_y_test = [1]
  print (model.evaluate(s_x_test, s_y_test)) 
  
  # g - showing model 
  print('>>> showing generated cnn model ...')
  print(model)
  
  # h - saving the cnn model
  print('>>> saving the cnn model ...')
  
  print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

# start of main program code 
# run_test_case()
