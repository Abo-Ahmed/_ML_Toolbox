class Vgg16(BasicModel): 

    def build (self):
        super().build()
        cnnBase = VGG16(   input_shape=(self.rows, self.columns, self.channels),
                            classes=1000,
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classifier_activation="softmax")

        self.model = Sequential()
        self.model.add(cnnBase)
        self.model.add(GlobalAveragePooling2D())

        # cnnOut = GlobalAveragePooling2D()(cnnBase.output)
        # cnnBase.trainable = False
        # self.model = Model(cnnBase.input, cnnOut)

        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


    def build_1 (self):
        super().build()
        channels, rows, columns = 3,512,512
        cnnBase = VGG16(   input_shape=(rows, columns, channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=1000,
                            classifier_activation="softmax")
        # cnnBase.trainable = False
        cnnOut = GlobalAveragePooling2D()(cnnBase.output)
        self.model = Model(cnnBase.input, cnnOut)

        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


