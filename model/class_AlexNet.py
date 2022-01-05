
class AlexNet(BasicModel):

    def build (self):
        super().build()
        
        # Initialize model
        self.model = tf.keras.models.Sequential()

        # Layer 1
        self.model.add(keras_layers.Conv2D(96, (11, 11), input_shape=img_shape,padding='same', kernel_regularizer=l2(l2_reg)))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))
        self.model.add(keras_layers.convolutional.MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        self.model.add(keras_layers.Conv2D(256, (5, 5), padding='same'))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))
        self.model.add(keras_layers.convolutional.MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        self.model.add(keras_layers.convolutional.ZeroPadding2D((1, 1)))
        self.model.add(keras_layers.Conv2D(512, (3, 3), padding='same'))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))
        self.model.add(keras_layers.convolutional.MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        self.model.add(keras_layers.convolutional.ZeroPadding2D((1, 1)))
        self.model.add(keras_layers.Conv2D(1024, (3, 3), padding='same'))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))

        # Layer 5
        self.model.add(keras_layers.convolutional.ZeroPadding2D((1, 1)))
        self.model.add(keras_layers.Conv2D(1024, (3, 3), padding='same'))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))
        self.model.add(keras_layers.convolutional.MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        self.model.add(keras_layers.Flatten())
        self.model.add(keras_layers.Dense(3072))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))
        self.model.add(keras_layers.Dropout(0.5))

        # Layer 7
        self.model.add(keras_layers.Dense(4096))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('relu'))
        self.model.add(keras_layers.Dropout(0.5))

        # Layer 8
        self.model.add(keras_layers.Dense(n_classes))
        self.model.add(keras_layers.normalization.BatchNormalization())
        self.model.add(keras_layers.Activation('softmax'))
        
        
        # self.model.compile(optimizer='adam',
        #                 loss='sparse_categorical_crossentropy',
        #                 metrics=['accuracy'])

