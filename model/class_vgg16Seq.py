class vgg16Seq(basic_model): 

    def build (self):
        super().build()
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=(512,512,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(256, activation='relu', name='fc1'))
        self.model.add(Dense(128, activation='relu', name='fc2'))
        self.model.add(Dense(1, activation='sigmoid', name='output'))

        opt = SGD(lr=1e-4, momentum=0.9)
        self.model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

    # anther implementation - not fully working
    # def build (self):
    #     super().build()
    #     self.model = Sequential()
    #     self.model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    #     self.model.add(Convolution2D(64, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(64, 3, 3, activation='relu'))
    #     self.model.add(MaxPooling2D((2,2), strides=(2,2) , padding='same'))

    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(128, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(128, 3, 3, activation='relu'))
    #     self.model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(256, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(256, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(256, 3, 3, activation='relu'))
    #     self.model.add(MaxPooling2D((2,2), strides=(2,2) , padding='same'))

    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(512, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(512, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(512, 3, 3, activation='relu'))
    #     self.model.add(MaxPooling2D((2,2), strides=(2,2) , padding='same'))

    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(512, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(512, 3, 3, activation='relu'))
    #     self.model.add(ZeroPadding2D((1,1)))
    #     self.model.add(Convolution2D(512, 3, 3, activation='relu'))
    #     self.model.add(MaxPooling2D((2,2), strides=(2,2) , padding='same'))

    #     self.model.add(Flatten())
    #     self.model.add(Dense(4096, activation='relu'))
    #     self.model.add(Dropout(0.5))
    #     self.model.add(Dense(4096, activation='relu'))
    #     self.model.add(Dropout(0.5))
    #     self.model.add(Dense(1000, activation='softmax'))
        
    #     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #     self.model.compile(optimizer=sgd, loss='categorical_crossentropy')



