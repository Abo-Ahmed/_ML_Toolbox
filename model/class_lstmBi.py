class lstmBi(basic_model): 

    def build (self):
        super().build()
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
        self.model.add(Bidirectional(LSTM(10)))
        self.model.add(Dense(5))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


