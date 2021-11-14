class vggLstm(basic_model): 

    def build (self):
        super().build()
        channels, rows, columns = 3,512,512
        SequenceLength = 3 
        nClasses = 1
        video = Input(shape=(SequenceLength, rows, columns,channels))
        cnn_base = VGG16(   input_shape=(rows, columns, channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=nClasses,
                            classifier_activation="softmax")
        cnn_out = GlobalAveragePooling2D()(cnn_base.output)
        cnn = Model(cnn_base.input, cnn_out)
        encoded_frames = TimeDistributed(cnn)(video)
        encoded_sequence = LSTM(nClasses , return_sequences=True)(encoded_frames)
        hidden_layer = Dense(nClasses, activation="relu")(encoded_sequence)
        outputs = Dense(nClasses, activation="softmax")(hidden_layer)
        self.model = Model(video, outputs)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        handler.train_x = dataset.batchizeData(handler.train_x , SequenceLength )
        handler.train_y = dataset.batchizeData(handler.train_y , SequenceLength )
        handler.test_x = dataset.batchizeData(handler.test_x , SequenceLength )
        handler.test_y = dataset.batchizeData(handler.test_y , SequenceLength )
        print(handler.train_x)
        print(handler.train_x.shape)

