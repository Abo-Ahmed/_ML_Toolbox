class lstmBi(basic_model): 

    def build (self):
        SequenceLength = 5 
        nClasses = 1
        super().build()
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(nClasses, return_sequences=True), input_shape=(SequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(nClasses)))
        self.model.add(Dense(nClasses))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        handler.train_x = self.batchizeData(dataset.train_x , SequenceLength )
        handler.train_y = self.batchizeData(dataset.train_y , SequenceLength )
        handler.test_x = self.batchizeData(dataset.test_x , SequenceLength )
        handler.test_y = self.batchizeData(dataset.test_y , SequenceLength )

        print(handler.train_x)
        print(handler.train_x.shape)


