
class VggLstm(BasicModel): 

    def build (self):
        super().build()
        self.sequenceLength = 3 
        self.nClasses = 1
        video = tf.keras.Input(shape=(self.sequenceLength, self.rows, self.columns,self.channels))
        cnnBase = VGG16(   input_shape=(self.rows, self.columns, self.channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=self.nClasses,
                            classifier_activation="softmax")
        cnnOut = keras_layers.GlobalAveragePooling2D()(cnnBase.output)
        cnn = tf.keras.Model(cnnBase.input, cnnOut)
        encodedFrames = TimeDistributed(cnn)(video)
        encodedSequence = keras_layers.LSTM(self.nClasses , return_sequences=True)(encodedFrames)
        hiddenLayer = keras_layers.Dense(self.nClasses, activation="relu")(encodedSequence)
        outputs = keras_layers.Dense(self.nClasses, activation="softmax")(hiddenLayer)
        self.model = tf.keras.Model(video, outputs)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.batchize_data()
