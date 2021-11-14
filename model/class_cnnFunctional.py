
class CnnFunctional(BasicModel): 

    def build (self):
        super().build()
        self.model = tf.keras.Sequential()
        inputs = Input(shape=(512, 512, 3))
        t = layers.Flatten() (inputs)
        t = layers.Dense(512, activation=tf.nn.relu) (t)
        t = layers.Dropout(0.2) (t)
        outputs = layers.Dense(10, activation=tf.nn.softmax) (t)
        
        self.model = Model(inputs , outputs)
        self.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])


