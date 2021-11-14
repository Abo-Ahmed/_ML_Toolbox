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
        encoded_sequence = LSTM(1)(encoded_frames)
        print("ch 3")
        hidden_layer = Dense(1, activation="relu")(encoded_sequence)
        print("ch 4")
        outputs = Dense(1, activation="softmax")(hidden_layer)
        print("ch 5")

        self.model = Model(video, outputs)
        print("ch 6")

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        handler.train_x = dataset.batchizeData(handler.train_x , frames )
        handler.train_y = dataset.batchizeData(handler.train_y , frames )
        handler.test_x = dataset.batchizeData(handler.test_x , frames )
        handler.test_y = dataset.batchizeData(handler.test_y , frames )

        print(handler.train_x)
        print(handler.train_x.shape)



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