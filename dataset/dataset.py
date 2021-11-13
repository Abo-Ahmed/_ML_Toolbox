

class dataset:

  @staticmethod
  def read_folder_images(destnation_path):
    images = [ cv2.imread(file) for indx , file in enumerate(glob.glob( destnation_path + "/*.jpg")) if indx < handler.data_limit]
    img_array = [ np.resize( img.shape  , (handler.image_width , handler.image_height , 3 )) for img in images ]
    return np.array(img_array)

  @staticmethod
  def prepare_matrix(true_set ,false_set , title):
    if(handler.colored):
      x_set =  np.random.rand(len(true_set)+len(false_set),handler.image_width,handler.image_height , 3 )
      x_set[0:len(true_set),:,:,:] = true_set
      x_set[len(true_set):len(true_set)+len(false_set),:,:,:] = false_set
    else:
      x_set =  np.random.rand(len(true_set)+len(false_set),handler.image_width,handler.image_height )
      x_set[0:len(true_set),:,:] = true_set
      x_set[len(true_set):len(true_set)+len(false_set),:,:] = false_set

    x_set = x_set / 255.0
    
    y_set = np.random.rand(len(true_set)+len(false_set))
    y_set[0:len(true_set)] = np.ones(len(true_set))
    y_set[len(true_set):len(true_set)+len(false_set)] = np.zeros(len(false_set))

    print( '>> {} dimensions: {} , {}'.format( title , x_set.shape, y_set.shape ))
    configure.print_line('=')
    return x_set , y_set

  @staticmethod
  def read_images(first_path , second_path , title):
      print(">> reading " + title + " dataset ...")
      print("first_path",first_path)
      print("second_path",second_path)
      print("title",title)

      return dataset.prepare_matrix( dataset.read_folder_images(first_path ) , dataset.read_folder_images(second_path ) , title)

  @staticmethod
  def read_resize_image(img):
    if (handler.colored):
      return np.resize( cv2.imread( img).shape  , (handler.image_width , handler.image_height , 3 )) 
    else:
      return np.resize( cv2.cvtColor(cv2.imread( img),cv2.COLOR_BGR2GRAY).shape  , (handler.image_width , handler.image_height ))

  @staticmethod
  def read_predict():
      print(">> reading PREDICT images ...")
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
  def read_special_image(path):
    return  np.resize( cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY).shape  , (handler.image_width , handler.image_height ))
    
  @staticmethod
  def read_random():
      print(">> reading RANDOM images ...")
      randomImg = random.choice(os.listdir("/content/_master/dataset/random"))
      handler.predict_x = np.array([dataset.read_resize_image("/content/_master/dataset/random/"+randomImg)])
      print( '>>> {}: {}'.format( 'random: ' , len(handler.predict_x)))
      print(handler.predict_x)
      configure.print_line('=')

  @staticmethod
  def upload_images():
    uploaded = files.upload()
    for fn in uploaded.keys():
      print('>> User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
      print(uploaded[fn])
    print(uploaded.keys())
    return list(uploaded.keys())
  

