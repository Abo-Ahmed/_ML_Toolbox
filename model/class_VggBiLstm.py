class VggBiLstm(BasicModel): 

    def build (self):
        super().build()
        sequenceLength = 3 

        video = Input(shape=(sequenceLength, self.rows, self.columns,self.channels))

        cnnBase = VGG16(   input_shape=(self.rows, self.columns, self.channels),
                            classes=self.nClasses, weights=None)
        # cnnOut = GlobalAveragePooling2D()(cnnBase.output)
        cnn = Model(cnnBase.input, cnnBase)

        self.model = Sequential()
        self.model.add(video)
        self.model.add(TimeDistributed(cnn))
        self.model.add(Bidirectional(LSTM(self.nClasses, return_sequences=True), input_shape=(sequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(self.nClasses), input_shape=(sequenceLength, 1)))
        self.model.add(Dense(self.nClasses, activation="relu"))
        self.model.add(Dense(self.nClasses, activation="softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if(not handler.batched):
            self.batchize_data(sequenceLength)
            handler.batched = True

    def build_1 (self):
        super().build()
        sequenceLength = 3 

        video = Input(shape=(sequenceLength, self.rows, self.columns,self.channels))

        cnnBase = VGG16(   input_shape=(self.rows, self.columns, self.channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=self.nClasses,
                            classifier_activation="softmax")
        cnnOut = GlobalAveragePooling2D()(cnnBase.output)
        cnn = Model(cnnBase.input, cnnOut)

        self.model = Sequential()
        self.model.add(video)
        self.model.add(TimeDistributed(cnn))
        self.model.add(Bidirectional(LSTM(self.nClasses, return_sequences=True), input_shape=(sequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(self.nClasses), input_shape=(sequenceLength, 1)))
        self.model.add(Dense(self.nClasses, activation="relu"))
        self.model.add(Dense(self.nClasses, activation="softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if(not handler.batched):
            self.batchize_data(sequenceLength)
            handler.batched = True

