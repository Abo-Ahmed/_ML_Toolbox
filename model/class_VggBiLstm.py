class VggBiLstm(BasicModel): 

    def build (self):
        super().build()
        self.sequenceLength = 3 

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(self.sequenceLength, self.rows, self.columns,self.channels)))
        self.model.add(TimeDistributed(
                            VGG16(   
                                input_shape=(self.rows, self.columns, self.channels),
                                classes=self.nClasses, weights=None)))
        self.model.add(keras_layers.Bidirectional(keras_layers.LSTM(self.nClasses, return_sequences=True), input_shape=(self.sequenceLength, 1)))
        self.model.add(keras_layers.Bidirectional(keras_layers.LSTM(self.nClasses), input_shape=(self.sequenceLength, 1)))
        self.model.add(keras_layers.Dense(self.nClasses, activation="relu"))
        self.model.add(keras_layers.Dense(self.nClasses, activation="softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.batchize_data()
