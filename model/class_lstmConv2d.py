class lstmConv2d(basic_model): 

    def build (self):
        super().build()
        channels, rows, columns = 3,512,512
        SequenceLength = 3 
        nClasses = 1
        in_shape = (SequenceLength, rows, columns, channels)
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
        # self.model.add(Reshape((nClasses, out_shape[1] * out_shape[2] * out_shape[3] * out_shape[4])))
        # self.model.add(LSTM(nClasses, return_sequences=False))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(nClasses, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        handler.train_x = dataset.batchizeData(handler.train_x , SequenceLength )
        handler.train_y = dataset.batchizeData(handler.train_y , SequenceLength )
        handler.test_x = dataset.batchizeData(handler.test_x , SequenceLength )
        handler.test_y = dataset.batchizeData(handler.test_y , SequenceLength )
        print(handler.train_x)
        print(handler.train_x.shape)