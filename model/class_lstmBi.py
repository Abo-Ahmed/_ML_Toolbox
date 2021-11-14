class LstmBi(BasicModel): 

    def build (self):
        sequenceLength = 5 
        nClasses = 1
        super().build()
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(nClasses, return_sequences=True), input_shape=(sequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(nClasses, return_sequences=True), input_shape=(sequenceLength, 1)))
        self.model.add(Dense(nClasses))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop' ,  metrics='accuracy')

        handler.train_x = [ 11 , 52 , 66 , 88 , 91 , 100 , 1.1 , 11 , 52 , 66 , 88 , 91 , 100 , 1.1 ]
        handler.train_y = [ 1 , 2  ,  1 , 2  , 1 ,  1  , 2 , 1 , 2  ,  1 , 2  , 1 ,  1  , 2]

        handler.test_x = [ 11 , 52 , 66 , 88 , 91 , 100 , 1.1 , 11 , 52 , 66 , 88 , 91 , 100 , 1.1 ]
        handler.test_y = [ 1 , 2  ,  1 , 2  , 1 ,  1  , 2 , 1 , 2  ,  1 , 2  , 1 ,  1  , 2]

        handler.train_x = dataset.batchize_data(handler.train_x , sequenceLength )
        handler.train_y = dataset.batchize_data(handler.train_y , sequenceLength )
        handler.test_x = dataset.batchize_data(handler.test_x , sequenceLength )
        handler.test_y = dataset.batchize_data(handler.test_y , sequenceLength )

        print(handler.train_x)
        print(handler.train_x.shape)


