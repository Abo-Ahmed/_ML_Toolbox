

class handler:

  test_x = []
  test_y = []
  train_x = []
  train_y = []
  predict_x = []
  predict_y = []
  data_limit = 20
  image_width = 512
  image_height = 512
  colored = True
  batched = False
  project_directory = ''

  current_network = None
  start_time = 0 
  model_report = []
 
  @staticmethod
  def load_modules():
    folders = ['/handler' , '/tools', '/dataset' , '/model']
    for foldername in folders:
      for filename in os.listdir(handler.project_directory + foldername):
        if filename.endswith('.py') and filename != 'handler.py':
          execfile(handler.project_directory + foldername + '/' + filename)
          print(">>> " + filename + " loadded ...")
    configure.printer(">>> all modules loaded ...")

  @staticmethod
  def intial_configurations(mount , details , path):
    handler.start_time = time.time()
    handler.project_directory = path
    handler.load_modules()
    configure.show_version(mount, details) 
    configure.configure_tensor()
    configure.printer(">>> intial configurations done...")

  @staticmethod
  def read_data(data_path = None , img_width = 512 , img_height = 512):
      handler.image_width = img_width
      handler.image_height = img_height
      if data_path == None:
        data_path = handler.project_directory + '/dataset'
      handler.test_x , handler.test_y = dataset.read_images(data_path + '/test/NSFW', data_path + '/test/SFW' , "TEST")
      handler.train_x , handler.train_y = dataset.read_images(data_path + '/train/NSFW', data_path + '/train/SFW' , "TRAIN")

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
      handler.model_report.append(case + " ==> successful ")
      print(">>> successful model: " + case )
      handler.current_network = None
    except Exception as e :
      handler.model_report.append( case + " ==> failed ")
      print("XXX Error in model: " + case , e)
  
  @staticmethod
  def special_run(model_name , program = None, load = None):
    handler.current_network = globals()[model_name]()
    if (load):
      handler.current_network.load_model(load)
    else:
      handler.current_network.build()
      handler.current_network.summery_plot()
    if(program):
      getattr(handler.current_network, program)()

  @staticmethod
  def final_configurations() :
    configure.print_line()
    print(">>> final results: \n", "\n".join(handler.model_report))
    configure.show_period(time.time() - handler.start_time)
    
print(">>> handler module loadded ...")
