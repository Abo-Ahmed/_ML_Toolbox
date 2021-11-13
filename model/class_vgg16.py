class vgg16(basic_model): 

    def build (self):
        super().build()
        channels, rows, columns = 3,512,512
        cnn_base = VGG16(   input_shape=(rows, columns, channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=1000,
                            classifier_activation="softmax")
        # cnn_base.trainable = False
        cnn_out = GlobalAveragePooling2D()(cnn_base.output)
        self.model = Model(cnn_base.input, cnn_out)

        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


