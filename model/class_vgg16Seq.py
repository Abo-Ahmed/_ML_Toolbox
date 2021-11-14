class vgg16Seq(BasicModel): 

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
