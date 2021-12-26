
from os import name


class CnnSeq(BasicModel): 

    def build (self):
        super().build()
        self.model = tf.keras.tf.keras.models.Sequential()
        self.model.add(keras_layers.Flatten(input_shape=(512, 512,3) , name="flatten_1"))
        self.model.add(keras_layers.Dense(512, activation=tf.nn.relu , name= "dense_2"))
        self.model.add(keras_layers.Dropout(0.2,name= "dropout_3"))
        self.model.add(keras_layers.Dense(10, activation=tf.nn.softmax , name = "dense_4"))
        
        self.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        
