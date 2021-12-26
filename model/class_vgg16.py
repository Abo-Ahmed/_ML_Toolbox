class Vgg16(BasicModel): 

    def build (self):
        super().build()

        self.model = tf.keras.models.Sequential()
        self.model.add(VGG16(   input_shape=(self.rows, self.columns, self.channels),
                            classes=self.nClasses, weights=None))

        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
        cnnOut = keras_layers.GlobalAveragePooling2D()(cnnBase.output)
        self.model = tf.keras.Model(cnnBase.input, cnnOut)

        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


