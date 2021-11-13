
class configure:

  @staticmethod
  def print_line (symbol = None):
    if symbol == None:
      print('%' * 40 + '\n')
    else:
      print(symbol * 30)

  @staticmethod
  def printer(txt,symbol = None):
    print(txt)
    configure.print_line(symbol)

  @staticmethod
  def checkTools(*tools):
      for tool in tools:
          print(os.popen('pip list -v | grep ' + tool).readlines())
          configure.print_line()

  @staticmethod
  def show_version (mount , details) :
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      #raise SystemError('GPU device not found')
      print('XX GPU device not found')
    print('>> Tenserflow version: ' + tf.__version__ + ' - Tenserflow Device Name: ' + device_name )
    print('>> Keras version: ' + tf.keras.__version__)
    
    if(mount):
      drive.mount('/content/drive')
    else:
      print(">> Drive already mounted... ")
    
    if(details):
      print('>> List of all local devices:')
      local_device_protos = device_lib.list_local_devices()
      print( [ x.name for x in local_device_protos ])
    configure.print_line()

  @staticmethod
  def configure_tensor() :
    print('>> tensor configuration ...')
    try:
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement  = True
      return config
    except:
      print('>> cannot configure tensorflow')
      return

  @staticmethod
  def readTags(filePath):
      text_file = open(os.path.abspath(os.getcwd()) + "/" + filePath, "r")
      lines = text_file.readlines()
      text_file.close()
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
    print("--- execution time: {} minutes , {} seconds ---".format(minutes  , seconds - (minutes * 60) ))


  
      