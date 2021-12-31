
  @staticmethod
  def read_folder(destnation_path):
    counter = 0 
    imageSet = []
    items = os.listdir(destnation_path)
    for img in items:
      if img.endswith(".jpg"):
        imageSet.append(dataset.read_resize_image(destnation_path + '/' + img))
        counter = counter + 1
        if (counter > 30):
          break
    print( '>>> {}: {}'.format( destnation_path , len(imageSet)))
    print(items)
    configure.print_line('=')
    items.clear()
    return np.array(imageSet)
