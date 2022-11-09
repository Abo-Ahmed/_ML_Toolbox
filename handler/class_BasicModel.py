
# fit() --> training , evaluate() --> testing , predict() --> prediction

class BasicModel (object):
    ################
    ## constructor and destructor
    ################
    def __init__(self):
        self.model = None
        self.name = type(self).__name__
        self.path = "/content/drive/MyDrive/eng-mahmoud/models/"
        self.loopEpochs = 30
        self.loopIndex = 0 
        self.loopLimit = 1000
        self.sequenceLength = 3

        self.loss = 0
        self.acc = 0
        self.result = results()

        self.channels = 3 if(handler.isColored) else 1
        self.rows = handler.imageWidth
        self.columns =  handler.imageHeight
        self.nClasses = 3

        print(">>> " + self.name + " model intiated ...")

    def __del__(self):
        print("XXX deleted, model " + self.name)

    ################
    ## model details
    ################
    def build (self):
        print('>>> bulding ' + self.name + ' model ...')

    def summery_plot(self , details = True, plotted = True):
        try:
            if(details):
                self.summery()
            if(plotted):
                self.plot()
        except Exception as e:
            print("XXX Error executing summery_plot model")
            
    def plot(self):
        print('>>> plotting model: ' + self.name)
        tf.keras.utils.plot_model(self.model, to_file=self.name + '.png', show_shapes=True)

    def summery(self):
        print('>>> showing ' + self.name + ' summery ...')
        self.model.summary()

    ################
    ## model operations
    ################
    def experiment_basic(self,ep = 5):
        self.train(ep)
        self.test()

    def experiment_loop(self , ep = 5 ):
        self.loop_train()

    def loop_train(self):
        self.load_parameters() 
        self.load_weights()
        for i in range(self.loopLimit - self.loopIndex):
            handler.batchNo += 1 
            dataset.read_real_data()
            handler.batched = False
            # self.batchize_data()
            self.train(self.loopEpochs)
            self.test()
            self.save_weights(title=("odd" if ((i + self.loopIndex ) % 2) else "even"))
            self.save_parameters(i + self.loopIndex)
            configure.print_line()
            if(i % 151 == 150):
                print(">>> DO you wish to continue? y / n")
                if(input() != 'y'):
                    break

    def experiment_confusion(self,ep = 5):
        self.train(ep)
        self.test()

        predicted_y = self.model.predict(handler.train_x)
        predicted = []
        for item in predicted_y:
            if(item[0] > item[1] and item[0] > item[2] ):
                predicted.append(0)
            elif (item[1] > item[0] and item[1] > item[2] ):
                predicted.append(1)
            else:
                predicted.append(2)
        results.confusion_matrix(handler.train_y , predicted)


    # verbose=0 --> (silent), 1 --> animated progress , 2 -->  mention epoch no. 
    def train (self , epochs = 5):
        print('>>> training ' + self.name + ' model ...')
        self.model.fit(np.array(handler.train_x), np.array(handler.train_y), epochs , verbose=1)

    def test (self):
        print('>>> testing ' + self.name + ' model ...')
        self.loss, self.acc = self.model.evaluate(np.array(handler.test_x), np.array(handler.test_y), verbose=2)

    def predict (self , smpl = None):
        print('>>> Predicting with ' + self.name + ' model ...')
        if smpl:
            handler.predict_x = smpl
        else:
            dataset.read_predict()
        handler.predict_y = self.model.predict(handler.predict_x)
        print('>>> predict_y:' , handler.predict_y)

    def inf_predict (self):
        while True :
            self.predict()

    def random_predict (self):
        print('>>> Random predict with ' + self.name + ' model ...')
        dataset.read_random()
        handler.predict_y = self.model.predict(handler.predict_x)
        print('>>> predict_y:' , handler.predict_y)

    def batchize_data(self,):
        if(not handler.batched):
            handler.train_x = dataset.batchize(handler.train_x , self.sequenceLength )
            handler.train_y = dataset.batchize(handler.train_y , self.sequenceLength )
            handler.test_x = dataset.batchize(handler.test_x , self.sequenceLength )
            handler.test_y = dataset.batchize(handler.test_y , self.sequenceLength )
            handler.batched = True
            print(">>> shape after batchizing: " , handler.train_x.shape)

    ################
    ## loading and saving model
    ################

    def load_model(self):
        print('>>> loading ' + self.name + ' model ...')
        try:
            self.model = tf.keras.models.load_model(self.path)
        except Exception as e:
            print('XXX failed loading model: ' + self.name  , e)

    def save_model(self):
        print('>>> saving ' + self.name + ' model ...')
        try:
            self.model.save(self.path) # saving the entire model
        except Exception as e:
            print('XXX failed saving ' + self.name + ' model ...' , e)

    def load_weights(self):
        print('>>> loading ' + self.name + ' weights ...')
        try:
            # odd only retrieved - even for backup
            self.model.load_weights(self.path + "odd-" + self.name + ".h5")
        except Exception as e:
            print('XXX failed loading weights for: ' + self.name  ,e)
            
    def checkpoint(self, checkpoint_path = "cp.ckpt"): # saves the best weights
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    def save_weights(self, title = ""): # saves the current weights
        print('>>> saving ' + self.name + ' weights ...')
        try:
            self.model.save_weights(self.path + title + "-" + self.name + ".h5" ) # saving weights only
        except Exception as e:
            print('XXX failed saving ' + self.name + ' weights ...' , e)

    def load_parameters(self):
        print('>>> loading ' + self.name + ' parameters ...')
        try:
            file = open(self.path + "params-" + self.name + ".pkl", "rb")
            dir = pickle.load(file)
            self.name = dir["name"]
            self.loopEpochs = int(dir["loopEpochs"])
            self.loopIndex = int(dir["loopIndex"]) 
            self.loopLimit = int(dir["loopLimit"])
            handler.batchNo = self.loopIndex
            print(self.name , self.loopEpochs , self.loopIndex , self.loopLimit)
        except Exception as e:
            print('XXX failed loading parameters for: ' + self.name  , e)

    def save_parameters(self,index):

        print(self.name , self.loopEpochs , index , self.loopLimit)
        dic =   {
                    "name": self.name,
                    "loopEpochs": self.loopEpochs ,
                    "loopIndex": index,
                    "loopLimit": self.loopLimit ,
                }
        file = open(self.path + "params-" + self.name + ".pkl" , "wb")
        pickle.dump(dic, file)
        file.close()        
