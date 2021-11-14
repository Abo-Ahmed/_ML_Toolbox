
class CnnSeq(BasicModel): 

    def build (self):
        super().build()
        self.model = tf.keras.Sequential()
        self.model.add(layers.Flatten(input_shape=(512, 512,3)))
        self.model.add(layers.Dense(512, activation=tf.nn.relu))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(10, activation=tf.nn.softmax))
        
        self.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        
