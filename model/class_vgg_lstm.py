class vggLstm(basic_model): 

    def build (self):
        super().build()
        frames, channels, rows, columns = 5,3,512,512
        video = Input(shape=(frames, rows, columns,channels))
        cnn_base = VGG16(   input_shape=(rows, columns, channels),
                            weights="imagenet", 
                            include_top=False ,                         
                            input_tensor=None,
                            pooling=None,
                            classes=1000,
                            classifier_activation="softmax")
        cnn_out = GlobalAveragePooling2D()(cnn_base.output)
        cnn = Model(cnn_base.input, cnn_out)
        print("ch 1")
        encoded_frames = TimeDistributed(cnn)(video)
        print("ch 2")
        encoded_sequence = LSTM(256)(encoded_frames)
        print("ch 3")
        hidden_layer = Dense(1024, activation="relu")(encoded_sequence)
        print("ch 4")
        outputs = Dense(10, activation="softmax")(hidden_layer)
        print("ch 5")

        self.model = Model(video, outputs)
        print("ch 6")

        self.model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    def program_1(self):
        self.model.fit(
            # Training data : features (review) and classes (positive or negative)
            handler.x_train, handler.y_train,
                            
            # Number of samples to work through before updating the 
            # internal model parameters via back propagation. The 
            # higher the batch, the more memory you need.
            batch_size=256, 

            # An epoch is an iteration over the entire training data.
            epochs=3, 
            
            # The model will set apart his fraction of the training 
            # data, will not train on it, and will evaluate the loss
            # and any model metrics on this data at the end of 
            # each epoch.
            validation_split=0.2,
            
            verbose=1
        )
        handler.y_test = self.model.predict_classes(handler.x_test)

        print(y_test)

    def train(self , epochs = 10):
        new_x = []
        new_y = []
        for i in range(len(handler.train_x)):
            if i % 5 == 0 and i != 0 and i != len(handler.train_x) :
                new_x.append(handler.train_x[i-5 : i])
                new_y.append(handler.train_y[i-5 : i])
        handler.train_x = new_x
        handler.train_y = new_y
        super().train(epochs)


# ValueError: in user code:

#     File "/usr/local/lib/python3.7/dist-packages/keras/engine/training.py", line 878, in train_function  *
#         return step_function(self, iterator)
#     File "/usr/local/lib/python3.7/dist-packages/keras/engine/training.py", line 867, in step_function  **
#         outputs = model.distribute_strategy.run(run_step, args=(data,))
#     File "/usr/local/lib/python3.7/dist-packages/keras/engine/training.py", line 860, in run_step  **
#         outputs = model.train_step(data)
#     File "/usr/local/lib/python3.7/dist-packages/keras/engine/training.py", line 808, in train_step
#         y_pred = self(x, training=True)
#     File "/usr/local/lib/python3.7/dist-packages/keras/utils/traceback_utils.py", line 67, in error_handler
#         raise e.with_traceback(filtered_tb) from None
#     File "/usr/local/lib/python3.7/dist-packages/keras/engine/input_spec.py", line 199, in assert_input_compatibility
#         raise ValueError(f'Layer "{layer_name}" expects {len(input_spec)} input(s),'

#     ValueError: Layer "model_4" expects 1 input(s), but it received 11 input tensors. 
#  Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:1' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:2' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:3' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:4' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:5' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:6' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:7' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:8' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:9' shape=(None, 512, 512, 3) dtype=float32>, 
# <tf.Tensor 'IteratorGetNext:10' shape=(None, 512, 512, 3) dtype=float32>]