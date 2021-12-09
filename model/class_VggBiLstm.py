class VggBiLstm(BasicModel): 

    def build (self):
        super().build()
        channels, rows, columns = 3,224,224
        sequenceLength = 3 
        nClasses = 1

        video = Input(shape=(sequenceLength, rows, columns,channels))

        cnnBase = VGG16(   input_shape=(rows, columns, channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=nClasses,
                            classifier_activation="softmax")
        cnnOut = GlobalAveragePooling2D()(cnnBase.output)
        cnn = Model(cnnBase.input, cnnOut)

        self.model = Sequential()
        self.model.add(video)
        self.model.add(TimeDistributed(cnn))
        self.model.add(Bidirectional(LSTM(nClasses, return_sequences=True), input_shape=(sequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(nClasses), input_shape=(sequenceLength, 1)))
        self.model.add(Dense(nClasses, activation="relu"))
        self.model.add(Dense(nClasses, activation="softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if(not handler.batched):
            handler.train_x = dataset.batchize_data(handler.train_x , sequenceLength )
            handler.train_y = dataset.batchize_data(handler.train_y , sequenceLength )
            handler.test_x = dataset.batchize_data(handler.test_x , sequenceLength )
            handler.test_y = dataset.batchize_data(handler.test_y , sequenceLength )
            print(handler.train_x)
            print(handler.train_x.shape)
            handler.batched = True

