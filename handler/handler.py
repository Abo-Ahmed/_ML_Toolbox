

class handler:

  test_x = []
  test_y = []
  train_x = []
  train_y = []
  predict_x = []
  predict_y = []

  dataSize = 100
  batchSize = 20
  imageWidth = 512
  imageHeight = 512
  colored = True
  batched = False
  projectDir = ''

  currentNetwork = None
  startTime = 0 
  modelReport = []
 
  @staticmethod
  def load_modules():
    folders = ['/handler' , '/tools', '/dataset' , '/model']
    for folderName in folders:
      for fileName in os.listdir(handler.projectDir + folderName):
        if fileName.endswith('.py') and fileName != 'handler.py':
          execfile(handler.projectDir + folderName + '/' + fileName)
          print(">>> " + fileName + " loadded ...")
    configure.printer(">>> all modules loaded ...")

  @staticmethod
  def intial_configurations(mount , details , path):
    handler.startTime = time.time()
    handler.projectDir = path
    handler.load_modules()
    configure.show_version(mount, details) 
    configure.configure_tensor()
    configure.printer(">>> intial configurations done...")

  @staticmethod
  def read_data(dataPath = None , imgWidth = 512 , imgHeight = 512):
      handler.imageWidth = imgWidth
      handler.imageHeight = imgHeight
      if dataPath == None:
        dataPath = handler.projectDir + '/dataset'
      handler.test_x , handler.test_y = dataset.read_images(dataPath + '/test/NSFW', dataPath + '/test/SFW' , "TEST")
      handler.train_x , handler.train_y = dataset.read_images(dataPath + '/train/NSFW', dataPath + '/train/SFW' , "TRAIN")

  @staticmethod
  def read_real_data(dataPath , imgWidth = 512 , imgHeight = 512):
      handler.imageWidth = imgWidth
      handler.imageHeight = imgHeight
      handler.train_x = dataset.read_folder_images(dataPath,0)
      handler.train_y = dataset.get_prediction_matrix(dataPath,0)

  @staticmethod
  def run_models(models):
    for m in models:
      handler.run_model(m,"program_0") # model , program

  @staticmethod
  def run_model(model_name , program = None, load = None):
    try:
      case = model_name + " with : " + program
      print("<>"*50 )
      print(">>> starting model: " +  case)
      handler.special_run(model_name , program , load )
      handler.modelReport.append(case + " ==> successful ")
      print(">>> successful model: " + case )
      handler.currentNetwork = None
    except Exception as e :
      handler.modelReport.append( case + " ==> failed " + str(e).split("\n")[0] )
      print("XXX Error in model: " + case , e)
  
  @staticmethod
  def special_run(model_name , program = None, load = None):
    handler.currentNetwork = globals()[model_name]()
    if (load):
      handler.currentNetwork.load_model(load)
    else:
      handler.currentNetwork.build()
      handler.currentNetwork.summery_plot()
    if(program):
      getattr(handler.currentNetwork, program)()

  @staticmethod
  def final_configurations() :
    configure.print_line()
    print(">>> final results: \n", "\n".join(handler.modelReport))
    configure.show_period(time.time() - handler.startTime)
    
print(">>> handler module loadded ...")
