
class configure:

  @staticmethod
  def print_line (symbol = None):
    if symbol:
      print(symbol * 40)      
    else:
      print('%' * 40 + '\n')

  @staticmethod
  def printer(txt,symbol = None):
    print(txt)
    configure.print_line(symbol)

  @staticmethod
  def check_tools(*tools):
      for tool in tools:
          print(os.popen('pip list -v | grep ' + tool).readlines())
          configure.print_line()

  @staticmethod
  def show_version (mount , details) :
    deviceName = tf.test.gpu_device_name()
    if deviceName != '/device:GPU:0':
      print('XXX GPU device not found')
      deviceName = "GPU not found"
    else:
      # print(gpu_info)
      print(tf.test.gpu_device_name())
    print('>>> Tenserflow version: ' + tf.__version__ + ' - with ' + deviceName )
    print('>>> Keras version: ' + tf.keras.__version__)
    
    # show gpu usage
    # gpu_info = '\n'.join(!nvidia-smi)
    # if gpu_info.find('failed') >= 0:
    #   print('XXX Not connected to a GPU')

    # show high ram usage
    # ram_gb = virtual_memory().total / 1e9
    # print('>>> Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    # if ram_gb < 20:
    #   print('XXX Not using a high-RAM runtime')
    # else:
      # print('>>> You are using a high-RAM runtime!')
      
    if(mount):
      drive._mount('/content/drive')
    else:
      print(">>> Drive already mounted... ")
    
    if(details):
      print('>>> List of all local devices:')
      localDeviceProtos = device_lib.list_local_devices()
      print( [ x.name for x in localDeviceProtos ])
    configure.print_line()

  @staticmethod
  def configure_tensor() :
    print('>>> tensor configuration ...')
    try:
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement  = True
      config.log_device_placement = True
      return config
    except:
      print('>>> cannot configure tensorflow')
      return -1

  @staticmethod
  def read_tags(filePath):
      textFile = open(os.path.abspath(os.getcwd()) + "/" + filePath, "r")
      lines = textFile.readlines()
      textFile.close()
      print (len(lines))
      print (lines)
      return lines

  @staticmethod
  def use_cpu (callable) :
    return configure.use_device(callable , "/cpu:0")

  @staticmethod
  def use_gpu (callable) :
    return configure.use_device(callable , "/gpu:0")

  @staticmethod
  def use_device (callable , device = "/cpu:0", title = "code") :
    start = time.time() 
    try:
      with tf.device(device):  
        print(">>> Running "  + title + " on " + device)
        callable()
        return 0
    except Exception as e:
        print("XXX Failed "  + title + " on " + device , e)
        return -1
    finally :
      configure.show_period(time.time() - start)

  @staticmethod
  def show_period(seconds):
    minutes = int(seconds / 60)
    print("--- execution time: {} minutes , {:0.5} seconds ---".format(minutes  , seconds - (minutes * 60) ))


  
      