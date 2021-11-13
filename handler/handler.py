

class handler:

  def __init__(self):
      test_x = []
      test_y = []
      train_x = []
      train_y = []
      predict_x = []
      predict_y = []
      data_limit = 30
      image_width = 512
      image_height = 512
      colored = True
      project_directory = ''
  test_x = []
  test_y = []
  train_x = []
  train_y = []
  predict_x = []
  predict_y = []
  data_limit = 30
  image_width = 512
  image_height = 512
  colored = True
  project_directory = ''

  current_network = None
  start_time = 0 
  model_report = []
 
  @staticmethod
  def reset_variables():
      handler.test_x = []
      handler.test_y = []
      handler.train_x = []
      handler.train_y = []
      handler.predict_x = []
      handler.predict_y = []
      handler.data_limit = 30
      handler.image_width = 512
      handler.image_height = 512
      handler.colored = True
      handler.project_directory = ''

      handler.current_network = None
      handler.start_time = 0 
      handler.model_report = []

  @staticmethod
  def load_modules():
    folders = ['/handler' , '/tools', '/dataset' , '/model']
    for foldername in folders:
      for filename in os.listdir(handler.project_directory + foldername):
        if filename.endswith('.py') and filename != 'handler.py':
          execfile(handler.project_directory + foldername + '/' + filename)
          print(">> " + filename + " loadded ...")
    configure.printer(">> all modules loaded ...")

  @staticmethod
  def intial_configurations(mount , details , path):
    handler.start_time = time.time()
    handler.project_directory = path
    print("path" , path)
    print("handler.project_directory" ,handler.project_directory)
    handler.load_modules()
    handler.reset_variables()
    configure.show_version(mount, details) 
    configure.configure_tensor()
    configure.printer(">> intial configurations done...")


  @staticmethod
  def read_data(data_path = None , img_width = 512 , img_height = 512):
      handler.image_width = img_width
      handler.image_height = img_height
      if data_path == None:
        data_path = handler.project_directory + '/dataset'
      handler.test_x , handler.test_y = dataset.read_images(data_path + '/test/NSFW', data_path + '/test/SFW' , "TEST")
      handler.train_x , handler.train_y = dataset.read_images(data_path + '/train/NSFW', data_path + '/train/SFW' , "TRAIN")

  @staticmethod
  def runModel(model_name , program = None, load = None):
    try:
      print()
      print("<>"*60 )
      print(">>> starting model: " + model_name + " with : " + program )
      handler.special_run(model_name , program , load )
      handler.model_report.append(model_name + " with : " + program + " --> successful ")
      print(">>> successful model: " + model_name + " with : " + program )
    except Exception as e :
      handler.model_report.append(model_name + " --> failed ")
      print("XXX Error in model: " + model_name + " with : " + program)
      print(e)
  
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
    seconds = time.time() - handler.start_time
    minutes = seconds / 60
    print("--- execution time: {} minutes , {} seconds ---".format(int(minutes)  , int(seconds - (minutes * 60)) ))
    

print(">>> handler module loadded ...")
