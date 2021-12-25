class VggBiLstm(BasicModel): 

    def build (self):
        super().build()
        sequenceLength = 3 

        self.model = Sequential()
        self.model.add(Input(shape=(sequenceLength, self.rows, self.columns,self.channels)))
        self.model.add(TimeDistributed(
                            VGG16(   
                                input_shape=(self.rows, self.columns, self.channels),
                                classes=self.nClasses, weights=None)))
        self.model.add(Bidirectional(LSTM(self.nClasses, return_sequences=True), input_shape=(sequenceLength, 1)))
        self.model.add(Bidirectional(LSTM(self.nClasses), input_shape=(sequenceLength, 1)))
        self.model.add(Dense(self.nClasses, activation="relu"))
        self.model.add(Dense(self.nClasses, activation="softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.batchize_data(sequenceLength)
