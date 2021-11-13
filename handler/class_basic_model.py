# fit() is for training the model with the given inputs (and corresponding training labels).
# evaluate() is for evaluating the already trained model using the validation (or test) data and the corresponding labels. Returns the loss value and metrics values for the model.
# predict() is for the actual prediction. It generates output predictions for the input samples.

class basic_model (object):
    ################
    ## constructor
    ################
    def __init__(self):
        self.model = None
        self.name = type(self).__name__
        self.load_path = ''
        self.save_path = ''
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.CM = []
        
        self.all_acc = []
        self.all_loss = []
        self.loss = 0
        self.acc = 0

        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

        print(">> " + self.name + " model intiated ...")

    ################
    ## destructor
    ################
    def __del__(self):
        print("XX deleted, model " + self.name)

    ################
    ## model retreival
    ################
    def build (self):
        print('>> bulding ' + self.name + ' model ...')

    def load_model(self , path = None):
        print('>> loading ' + self.name + ' model ...')
        if path != None:
            self.load_path = path
        self.model = tf.keras.models.load_model(self.load_path)

    def load_weights(self , path = None):
        print('>> loading ' + self.name + ' weights ...')
        if path != None:
            self.load_path = path
        self.model.load_weights(self.load_path)

    ################
    ## model operations
    ################

    def program_0(self,ep = 10):
        self.train(ep)
        self.test()


    def train (self , epochs = 10):
        print('>> training ' + self.name + ' model ...')
        self.model.fit(handler.train_x, handler.train_y, epochs , verbose=1)

    # verbose=0 will show you nothing (silent)
    # verbose=1 will show you an animated progress
    # verbose=2 will just mention the number of epoch 
    def test (self):
        print('>> testing ' + self.name + ' model ...')
        self.loss, self.acc = self.model.evaluate(handler.test_x, handler.test_y, verbose=2)
        print('>>> Restored model, accuracy: {:5.2f}%'.format(100 * self.acc))

    def predict (self , smpl = None):
        dataset.read_predict()
        print('>> Predicting with ' + self.name + ' model ...')
        if smpl != None:
            handler.predict_x = smpl
        handler.predict_y = self.model.predict(handler.predict_x)
        print('>>> predict_y:')
        print(handler.predict_y)

    def inf_predict (self):
        print('>> Predicting with ' + self.name + ' model ...')
        while True :
            dataset.read_predict()
            handler.predict_y = self.model.predict(handler.predict_x)
            print('>>> predict_y:')
            print(handler.predict_y)

    def random_predict (self):
        print('>> Predicting with ' + self.name + ' model ...')
        dataset.read_random()
        handler.predict_y = self.model.predict(handler.predict_x)
        print('>>> predict_y:')
        print(handler.predict_y)

    ################
    ## saving model
    ################
    def save_weights(self, path = None): # saves the current weights
        if self.model == None:
            print("XX No modle found for " + self.name)
            return
        if path != None:
            self.save_path = path
        print('>> saving ' + self.name + ' weights ...')
        self.model.save_weights(self.save_path) # saving weights only

    def save_model(self, path = None):
        if self.model == None:
            print("XX No modle found for " + self.name)
            return
        if path != None:
            self.save_path = path
        print('>> saving ' + self.name + ' model ...')
        self.model.save(self.save_path) # saving the entire model

    def checkpoint(self, path = "cp.ckpt"): # saves the best weights
        if self.model == None:
            print("XX No modle found for " + self.name)
            return
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    ################
    ## model detials
    ################
    
    def summery_plot(self , details = True,plotted = True):
        if self.model == None:
            print("XX No modle found for " + self.name)
            return
        try:
            if(details):
                self.summery()
            if(plotted):
                self.plot()
        except Exception as e:
            print("Error plotting  mode")
        
    
    def plot(self):
        print('>> plotting model: ' + self.name)
        tf.keras.utils.plot_model(self.model, to_file=self.name + '.png', show_shapes=True)

    def summery(self):
        print('>> showing ' + self.name + ' summery ...')
        self.model.summary()

    ################
    ## accuracy
    ################

    def calculate_CM():
        self.CM = confusion_matrix(y_actu, y_pred)

    def calculate_results(self):
        [self.accuracy , self.precision , self.recall , self.f1] = results.get_results(self.TP , self.TN , self.FP , self.FN )

    def show_results(self):
        results.print_results(self.TP , self.TN , self.FP , self.FN )

    def show_graphs(self):
        graphs.plot_array(self.all_acc)
        graphs.plot_array(self.all_loss)
