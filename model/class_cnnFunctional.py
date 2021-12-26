
from os import name

class CnnFunctional(BasicModel): 

    def build (self):
        super().build()
        self.model = tf.keras.models.Sequential()
        inputs = tf.keras.Input(shape=(handler.imageWidth, handler.imageHeight, 3), name="input_1")
        t = keras_layers.Flatten(name="flatten_2") (inputs)
        t = keras_layers.Dense(handler.imageWidth, activation=tf.nn.relu , name="dense_3") (t)
        t = keras_layers.Dropout(0.2,name="dropout_4") (t)
        outputs = keras_layers.Dense(10, activation=tf.nn.softmax,name="dense_5") (t)
        
        self.model = tf.keras.Model(inputs , outputs)
        self.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])


