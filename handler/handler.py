
class handler:

  test_x = []
  test_y = []
  train_x = []
  train_y = []
  predict_x = []
  predict_y = []

  batched = False
  batchNo = 0 
  dataPath = ''
  batchSize = 30
  imageWidth = 512
  imageHeight = 512
  imageDepth = 3 
  isColored = True

  projectDir = ''
  currentNetwork = None
  startTime = 0 
  modelReport = []
 
  danbooruPath ='/content/drive/MyDrive/eng-mahmoud/dataSet/danbooru2019/images'
  generalPath = danbooruPath + '/original/0000' 
  trainPath = danbooruPath + '/train' 
  ePath = danbooruPath + '/explicit' 
  qPath = danbooruPath + '/questionable'
  sPath = danbooruPath + '/safe'

  pornography2k = '/content/drive/MyDrive/eng-mahmoud/dataSet/pornography-2k'
  pFrames = pornography2k + '/frames/vPorn'
  NEFrames = pornography2k + '/frames/vNonPornEasy'
  NDFrames = pornography2k + '/frames/vNonPornDifficulty'

  pVideos = pornography2k + '/videos/vPorn'
  NEVideos = pornography2k + '/videos/vNonPornEasy'
  NDVideos = pornography2k + '/videos/vNonPornDifficulty'

  img512Path = '/content/drive/MyDrive/eng-mahmoud/dataSet/danbooru2019/images/512px/0000'

  fileList = []

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
  def intial_config(mount , details , path):
    handler.startTime = time.time()
    handler.projectDir = path
    handler.load_modules()
    configure.show_version(mount, details) 
    configure.configure_tensor()
    configure.printer(">>> intial config done...")

  @staticmethod
  def dataset_config( path , imgWidth  , imgHeight , batchSize , startBatch,  isColored = True):
      print(">>> train data path: " +  path)
      handler.dataPath = path
      handler.imageWidth = imgWidth
      handler.imageHeight = imgHeight
      handler.batchSize = batchSize
      handler.batchNo = startBatch
      handler.isColored = isColored
    
  @staticmethod
  def read_sample(dataPath = None ):
      if dataPath == None:
        dataPath = handler.projectDir + '/dataset'
      handler.test_x , handler.test_y = dataset.read_sample_images(dataPath + '/test/NSFW', dataPath + '/test/SFW' , "TEST")
      handler.train_x , handler.train_y = dataset.read_sample_images(dataPath + '/train/NSFW', dataPath + '/train/SFW' , "TRAIN")

  @staticmethod
  def read_real(batchSize = 30 ):
      handler.batchSize = batchSize
      dataset.read_real_data()
      dataset.randomize_data(handler.train_x , handler.train_y)
      dataset.randomize_data(handler.test_x , handler.test_y)

  @staticmethod
  def run_models(models):
    for m in models:
      handler.run_model(m,"experiment_basic") # model , program

  @staticmethod
  def run_model(model_name , program = None, epoches = 5 , load = None):
    try:
      case = model_name + " with : " + program
      print("<>"*50 )
      print(">>> starting model: " +  case)
      handler.special_run(model_name , program , epoches ,  load )
      handler.modelReport.append(case + " ==> successful ")
      print(">>> successful model: " + case )
      handler.currentNetwork = None
    except Exception as e :
      handler.modelReport.append( case + " ==> failed " + str(e).split("\n")[0] )
      print("XXX Error in model: " + case , e)
  
  @staticmethod
  def special_run(model_name , program = None, epoches = 5 , load = None):
    handler.currentNetwork = globals()[model_name]()
    if (load):
      handler.currentNetwork.load_model(load)
    else:
      handler.currentNetwork.build()
      handler.currentNetwork.summery_plot()
    if(program):
      getattr(handler.currentNetwork, program)(epoches)

  @staticmethod
  def final_config() :
    configure.print_line()
    print(">>> final results: \n", "\n".join(handler.modelReport))
    configure.show_period(time.time() - handler.startTime)
    del handler.currentNetwork
    
print(">>> handler module loadded ...")
