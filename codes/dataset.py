

class dataset:

  ##############################
  ## real data methods
  ##############################  

  @staticmethod
  def read_folder_images(destnation_path):
    batchStart = handler.batchNo * handler.batchSize
    batchEnd = batchStart + handler.batchSize

    images = [ cv2.imread(file) for indx , file in enumerate(glob.glob( destnation_path + "/*.jpg")) if indx > batchStart and indx < batchEnd]
    if (handler.isColored):
      img_array = [ np.resize( img.shape  , (handler.imageWidth , handler.imageHeight , 3 )) for img in images ]
    else:
      img_array = [ np.resize( cv2.cvtColor( img ,cv2.COLOR_BGR2GRAY).shape   , (handler.imageWidth , handler.imageHeight  )) for img in images ]
    return np.array(img_array)

  @staticmethod
  def get_rating(destnation_path ):
    batchStart = handler.batchNo * handler.batchSize
    batchEnd = batchStart + handler.batchSize
    pred = [ rating.values[int(file.replace(".jpg","").split("/")[-1])] for indx , file in enumerate(glob.glob( destnation_path + "/*.jpg")) if indx > batchStart and indx < batchEnd]
    return pred

  @staticmethod
  def read_real_data():

    handler.train_x = dataset.read_folder_images(handler.dataPath)
    handler.train_y = dataset.get_rating(handler.dataPath)

    handler.batchNo += 1
    handler.test_x = dataset.read_folder_images(handler.dataPath)
    handler.test_y = dataset.get_rating(handler.dataPath)
    handler.batchNo -= 1

    print(">>> batch: " + str(handler.batchNo) + " - train_x shape:" + str(handler.train_x.shape)  , handler.train_y)

  @staticmethod
  def randomize_data( x , y ):
    fixed_seed = random.random()
    random.Random(fixed_seed).shuffle(x)
    random.Random(fixed_seed).shuffle(y)

  @staticmethod
  def batchize( dataList ,  batchSize ):
      batches = []
      for i in range(len(dataList) // batchSize ):
          batches.append(dataList[i * batchSize:i * batchSize + batchSize])
      # return np.ndarray(batches)
      return tf.convert_to_tensor(batches, dtype=tf.float32)
      # return batches

  ##############################
  ## predict data methods
  ##############################  

  @staticmethod
  def read_resize_image(img):
    if (handler.isColored):
      return np.resize( cv2.imread( img).shape  , (handler.imageWidth , handler.imageHeight , 3 )) 
    else:
      return np.resize( cv2.cvtColor(cv2.imread( img),cv2.COLOR_BGR2GRAY).shape  , (handler.imageWidth , handler.imageHeight ))

  @staticmethod
  def read_predict():
      print(">>> reading PREDICT images ...")
      imageSet = []
      items = dataset.upload_images()
      for img in items:
        if img.endswith(".jpg"):
          imageSet.append(dataset.read_resize_image('/content/' + img))
      handler.predict_x = np.array(imageSet)
      print( '>>> {}: {}'.format( 'predict: ' , len(handler.predict_x)))
      print(handler.predict_x)
      configure.print_line('=')

  @staticmethod
  def read_random():
      print(">>> reading RANDOM images ...")
      randomImg = random.choice(os.listdir("/content/_master_network/dataset/random"))
      handler.predict_x = np.array([dataset.read_resize_image("/content/_master_network/dataset/random/"+randomImg)])
      print( '>>> {}: {}'.format( 'random: ' , len(handler.predict_x)))
      print(handler.predict_x)
      configure.print_line('=')

  @staticmethod
  def upload_images():
    uploaded = files.upload()
    for fn in uploaded.keys():
      print('>>> User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
      print(uploaded[fn])
    print(uploaded.keys())
    return list(uploaded.keys())
  
  ##############################
  ## sample data methods
  ##############################  
  @staticmethod
  def prepare_sample(true_set ,false_set , title):
    if(handler.isColored):
      x_set =  np.random.rand(len(true_set)+len(false_set),handler.imageWidth,handler.imageHeight , 3 )
      x_set[0:len(true_set),:,:,:] = true_set
      x_set[len(true_set):len(true_set)+len(false_set),:,:,:] = false_set
    else:
      x_set =  np.random.rand(len(true_set)+len(false_set),handler.imageWidth,handler.imageHeight )
      x_set[0:len(true_set),:,:] = true_set
      x_set[len(true_set):len(true_set)+len(false_set),:,:] = false_set

    x_set = x_set / 255.0
    
    y_set = np.random.rand(len(true_set)+len(false_set))
    y_set[0:len(true_set)] = np.ones(len(true_set))
    y_set[len(true_set):len(true_set)+len(false_set)] = np.zeros(len(false_set))

    print( '>>> {} dimensions: {} , {}'.format( title , x_set.shape, y_set.shape ))
    configure.print_line('=')
    return x_set , y_set

  @staticmethod
  def read_sample_images(first_path , second_path , title):
      print(">>> reading sample" + title + " dataset ...")
      return dataset.prepare_sample( dataset.read_folder_images(first_path ) , dataset.read_folder_images(second_path ) , title)
