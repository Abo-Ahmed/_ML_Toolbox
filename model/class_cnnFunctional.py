
from os import name


class CnnFunctional(BasicModel): 

    def build (self):
        super().build()
        self.model = tf.keras.Sequential()
        inputs = Input(shape=(512, 512, 3), name="input_1")
        t = layers.Flatten(name="flatten_2") (inputs)
        t = layers.Dense(512, activation=tf.nn.relu , name="dense_3") (t)
        t = layers.Dropout(0.2,name="dropout_4") (t)
        outputs = layers.Dense(10, activation=tf.nn.softmax,name="dense_5") (t)
        
        self.model = Model(inputs , outputs)
        self.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])


