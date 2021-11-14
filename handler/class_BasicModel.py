
# fit() --> training , evaluate() --> testing , predict() --> prediction

class BasicModel (object):
    ################
    ## constructor and destructor
    ################
    def __init__(self):
        self.model = None
        self.name = type(self).__name__
        self.path = "/content/drive/MyDrive/CoLab/models/"
        self.loopEpochs = 15
        self.loopIndex = 0 
        self.loopLimit = 100
        self.loss = 0
        self.acc = 0
        self.result = results()
        print(">>> " + self.name + " model intiated ...")

    def __del__(self):
        print("XXX deleted, model " + self.name)

    ################
    ## model construction
    ################
    def build (self):
        print('>>> bulding ' + self.name + ' model ...')

    def load_model(self):
        print('>>> loading ' + self.name + ' model ...')
        try:
            self.model = tf.keras.models.load_model(self.path)
        except Exception as e:
            print('XXX failed loading model: ' + self.name  , e)

    def load_weights(self):
        print('>>> loading ' + self.name + ' weights ...')
        try:
            # odd only retrieved - even for backup
            self.model.load_weights(self.path + "odd-" + self.name + ".index")
        except Exception as e:
            print('XXX failed loading weights for: ' + self.name  ,e)
            
    def load_parameters(self):
        print('>>> loading ' + self.name + ' parameters ...')
        try:
            file = open(self.path + "params-" + self.name + ".pkl", "rb")
            dir = pickle.load(file)
            self.name = dir["name"]
            self.path = dir["path"]
            self.loopEpochs = int(dir["loopEpochs"])
            self.loopIndex = int(dir["loopIndex"]) 
            self.loopLimit = int(dir["loopLimit"])
        except Exception as e:
            print('XXX failed loading parameters for: ' + self.name  , e)

    ################
    ## model operations
    ################
    def program_0(self,ep = 5):
        self.train(ep)
        self.test()

    def program_1(self):
        self.loop_train()

    def loop_train(self):
        self.load_weights()
        self.load_parameters() 
        for i in range(self.loopLimit - self.loopIndex):
            dataset.load_new_batch(i + self.loopIndex) # todo: implement
            self.train(self.loopEpochs)
            self.test()
            self.save_weights(title=("odd" if ((i + self.loopIndex ) % 2) else "even"))
            self.save_parameters(i + self.loopIndex)
            configure.print_line()

    # verbose=0 --> (silent), 1 --> animated progress , 2 -->  mention epoch no. 
    def train (self , epochs = 5):
        print('>>> training ' + self.name + ' model ...')
        self.model.fit(handler.train_x, handler.train_y, epochs , verbose=1)

    def test (self):
        print('>>> testing ' + self.name + ' model ...')
        self.loss, self.acc = self.model.evaluate(handler.test_x, handler.test_y, verbose=2)
        print('>>> current accuracy: {:5.2f}%'.format(100 * self.acc))

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

    ################
    ## saving model
    ################
    def save_parameters(self,index):
        print('>>> saving ' + self.name + ' parameters with index: ' + str(index))
        dic =   {
                    "name": self.name,
                    "path": self.path ,
                    "path": self.path,
                    "loopEpochs": self.loopEpochs ,
                    "loopIndex": self.loopIndex,
                    "loopLimit": index ,
                    "acc": self.acc,
                    "loss": self.loss 
                }
        file = open(self.path + "params-" + self.name + ".pkl" , "wb")
        pickle.dump(dic, file)
        file.close()        
        
    def save_weights(self, title = ""): # saves the current weights
        print('>>> saving ' + self.name + ' weights ...')
        try:
            self.model.save_weights(self.path + title + "-" + self.name ) # saving weights only
        except Exception as e:
            print('XXX failed saving ' + self.name + ' weights ...' , e)

    def save_model(self):
        print('>>> saving ' + self.name + ' model ...')
        try:
            self.model.save(self.path) # saving the entire model
        except Exception as e:
            print('XXX failed saving ' + self.name + ' model ...' , e)

    def checkpoint(self, checkpoint_path = "cp.ckpt"): # saves the best weights
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    ################
    ## model detials
    ################
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
