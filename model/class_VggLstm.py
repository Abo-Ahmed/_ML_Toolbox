
class VggLstm(BasicModel): 

    def build (self):
        super().build()
        self.sequenceLength = 3 
        self.nClasses = 1
        video = Input(shape=(self.sequenceLength, self.rows, self.columns,self.channels))
        cnnBase = VGG16(   input_shape=(self.rows, self.columns, self.channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=self.nClasses,
                            classifier_activation="softmax")
        cnnOut = GlobalAveragePooling2D()(cnnBase.output)
        cnn = Model(cnnBase.input, cnnOut)
        encodedFrames = TimeDistributed(cnn)(video)
        encodedSequence = LSTM(self.nClasses , return_sequences=True)(encodedFrames)
        hiddenLayer = Dense(self.nClasses, activation="relu")(encodedSequence)
        outputs = Dense(self.nClasses, activation="softmax")(hiddenLayer)
        self.model = Model(video, outputs)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.batchize_data()
