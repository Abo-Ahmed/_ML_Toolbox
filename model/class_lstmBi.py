class LstmBi(BasicModel): 

    def build (self):
        self.sequenceLength = 5 
        self.nClasses = 1
        super().build()
        self.model = tf.keras.models.Sequential()
        self.model.add(keras_layers.Bidirectional(keras_layers.LSTM(self.nClasses, return_sequences=True), input_shape=(self.sequenceLength, 1)))
        self.model.add(keras_layers.Bidirectional(keras_layers.LSTM(self.nClasses, return_sequences=True), input_shape=(self.sequenceLength, 1)))
        self.model.add(keras_layers.Dense(self.nClasses))
        self.model.add(keras_layers.LSTM('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop' ,  metrics='accuracy')

        handler.train_x = [ 11 , 52 , 66 , 88 , 91 , 100 , 1.1 , 11 , 52 , 66 , 88 , 91 , 100 , 1.1 ]
        handler.train_y = [ 1 , 2  ,  1 , 2  , 1 ,  1  , 2 , 1 , 2  ,  1 , 2  , 1 ,  1  , 2]

        handler.test_x = [ 11 , 52 , 66 , 88 , 91 , 100 , 1.1 , 11 , 52 , 66 , 88 , 91 , 100 , 1.1 ]
        handler.test_y = [ 1 , 2  ,  1 , 2  , 1 ,  1  , 2 , 1 , 2  ,  1 , 2  , 1 ,  1  , 2]

        self.batchize_data()


