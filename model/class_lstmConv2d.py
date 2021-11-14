
class LstmConv2d(BasicModel): 

    def build (self):
        super().build()
        channels, rows, columns = 3,512,512
        sequenceLength = 2 
        nClasses = 1
        nNodes = 32 # originally it was 32
        in_shape = (sequenceLength, rows, columns, channels)
        self.model = Sequential()
        self.model.add(ConvLSTM2D(nNodes, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.model.add(ConvLSTM2D(nNodes * 2, kernel_size=(5, 5), padding='valid', return_sequences=True))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.model.add(ConvLSTM2D(nNodes * 3, kernel_size=(3, 3), padding='valid', return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(ConvLSTM2D(nNodes * 3, kernel_size=(3, 3), padding='valid', return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(ConvLSTM2D(nNodes * 3, kernel_size=(3, 3), padding='valid', return_sequences=True))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.model.add(Dense(320))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        outShape = self.model.output_shape
        print(outShape)
        self.model.add(Reshape((sequenceLength,  outShape[2] * outShape[3] * outShape[4])))
        self.model.add(LSTM(sequenceLength, return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(sequenceLength, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        if(not handler.batched):
            handler.train_x = dataset.batchize_data(handler.train_x , sequenceLength )
            handler.train_y = dataset.batchize_data(handler.train_y , sequenceLength )
            handler.test_x = dataset.batchize_data(handler.test_x , sequenceLength )
            handler.test_y = dataset.batchize_data(handler.test_y , sequenceLength )
            print(handler.train_x)
            print(handler.train_x.shape)
            handler.batched = True