
class cnnSeq(basic_model): 

    def build (self):
        super().build()
        print("ch 1")
        self.model = tf.keras.Sequential()
        print("ch 2")
        self.model.add(layers.Flatten(input_shape=(512, 512,3)))
        print("ch 3")
        self.model.add(layers.Dense(512, activation=tf.nn.relu))
        print("ch 4")
        self.model.add(layers.Dropout(0.2))
        print("ch 5")
        self.model.add(layers.Dense(10, activation=tf.nn.softmax))
        print("ch 6")
        
        self.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        print("ch 7")

