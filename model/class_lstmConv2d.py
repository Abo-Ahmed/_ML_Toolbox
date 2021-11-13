class lstmConv2d(basic_model): 

    SequenceLength = 3 
    N_CLASSES = 3

    def build (self):
        super().build()
        in_shape = (self.SequenceLength, handler.image_width, handler.image_height, 3)
        self.model = Sequential()
        self.model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.model.add(ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.model.add(Dense(320))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        out_shape = self.model.output_shape
        print(out_shape)
        self.model.add(Reshape((self.SequenceLength, out_shape[2] * out_shape[3] * out_shape[4])))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.N_CLASSES, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        handler.train_x = self.batchizeData(handler.train_x , self.SequenceLength )
        handler.train_y = self.batchizeData(handler.train_y , self.SequenceLength )
        handler.test_x = self.batchizeData(handler.test_x , self.SequenceLength )
        handler.test_y = self.batchizeData(handler.test_y , self.SequenceLength )
        
    def batchizeData(self , dataList ,  batchSize ):
        batches = []
        for i in range(len(dataList) // batchSize ):
            batches.append(dataList[i * batchSize:i * batchSize + batchSize])
        return np.ndarray(batches)
        # return batches
