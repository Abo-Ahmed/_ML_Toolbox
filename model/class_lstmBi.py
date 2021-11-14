class lstmBi(basic_model): 

    def build (self):
        SequenceLength = 5 
        nClasses = 1
        super().build()
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(nClasses, return_sequences=True), input_shape=(SequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(nClasses, return_sequences=True), input_shape=(SequenceLength, 1)))
        self.model.add(Dense(nClasses))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop' ,  metrics='accuracy')

        handler.train_x = [ 11 , 52 , 66 , 88 , 91 , 100 , 1.1 , 11 , 52 , 66 , 88 , 91 , 100 , 1.1 ]
        handler.train_y = [ 1 , 2  ,  1 , 2  , 1 ,  1  , 2 , 1 , 2  ,  1 , 2  , 1 ,  1  , 2]

        handler.test_x = [ 11 , 52 , 66 , 88 , 91 , 100 , 1.1 , 11 , 52 , 66 , 88 , 91 , 100 , 1.1 ]
        handler.test_y = [ 1 , 2  ,  1 , 2  , 1 ,  1  , 2 , 1 , 2  ,  1 , 2  , 1 ,  1  , 2]

        handler.train_x = dataset.batchizeData(handler.train_x , SequenceLength )
        handler.train_y = dataset.batchizeData(handler.train_y , SequenceLength )
        handler.test_x = dataset.batchizeData(handler.test_x , SequenceLength )
        handler.test_y = dataset.batchizeData(handler.test_y , SequenceLength )

        print(handler.train_x)
        print(handler.train_x.shape)


