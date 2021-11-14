class Vgg16(BasicModel): 

    def build (self):
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


