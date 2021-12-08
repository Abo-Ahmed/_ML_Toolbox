class VggBiLstm(BasicModel): 

    def build (self):
        super().build()
        channels, rows, columns = 3,224,224
        sequenceLength = 3 
        nClasses = 3
        sFilter = 512

        self.model = Sequential()
        self.model.add(Conv2D(input_shape=(rows,columns,channels),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=sFilter / 4, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter / 4, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=sFilter / 2, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter / 2, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter / 2, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=sFilter, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=sFilter, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=sFilter, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(sFilter / 2, activation='relu', name='fc1'))
        self.model.add(Dense(sFilter / 4, activation='relu', name='fc2'))
        # self.model.add(Dense(1, activation='sigmoid', name='output')) # old
        self.model.add(Dropout(0.5)) # new


        outShape = self.model.output_shape
        print(outShape)
        self.model.add(Reshape((sequenceLength,  outShape[2] * outShape[3] * outShape[4])))
        self.model.add(LSTM(sequenceLength, return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(sequenceLength, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # opt = SGD(lr=1e-4, momentum=0.9)
        # self.model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])



        # if(not handler.batched):
        #     handler.train_x = dataset.batchize_data(handler.train_x , sequenceLength )
        #     handler.train_y = dataset.batchize_data(handler.train_y , sequenceLength )
        #     handler.test_x = dataset.batchize_data(handler.test_x , sequenceLength )
        #     handler.test_y = dataset.batchize_data(handler.test_y , sequenceLength )
        #     print(handler.train_x)
        #     print(handler.train_x.shape)
        #     handler.batched = True

    def build_2 (self):
        super().build()
        channels, rows, columns = 3,224,224
        sequenceLength = 3 
        nClasses = 3
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
        encodedFrames = TimeDistributed(cnn)(video)
        encodedSequence = LSTM(nClasses , return_sequences=True)(encodedFrames) # old
        # encodedSequence = Bidirectional(LSTM(nClasses , return_sequences=True), input_shape=(sequenceLength, 1))(encodedFrames) # new
        # encodedSequence = Bidirectional(LSTM(nClasses , return_sequences=True), input_shape=(sequenceLength, 1))(encodedSequence) # new

        hiddenLayer = Dense(nClasses, activation="relu")(encodedSequence)
        outputs = Dense(nClasses, activation="softmax")(hiddenLayer)
        self.model = Model(video, outputs)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if(not handler.batched):
            handler.train_x = dataset.batchize_data(handler.train_x , sequenceLength )
            handler.train_y = dataset.batchize_data(handler.train_y , sequenceLength )
            handler.test_x = dataset.batchize_data(handler.test_x , sequenceLength )
            handler.test_y = dataset.batchize_data(handler.test_y , sequenceLength )
            print(handler.train_x)
            print(handler.train_x.shape)
            handler.batched = True