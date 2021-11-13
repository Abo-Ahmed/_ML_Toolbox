class lstm(basic_model): 

    def build (self):
        super().build()
        self.model = Sequential()
        temp = tf.keras.layers.LSTM(
            units = 10,
            input_shape=( 5 , 512 ,512),
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            time_major=False,
            unroll=False )

        self.model.add(temp)
        self.model.add(Dense(5))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



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
#     File "/usr/local/lib/python3.7/dist-packages/keras/engine/input_spec.py", line 263, in assert_input_compatibility
#         raise ValueError(f'Input {input_index} of layer "{layer_name}" is '

#     ValueError: Input 0 of layer "sequential_4" is incompatible with the 
#      layer: expected shape=(None, 5, 10), found shape=(10, 512, 512, 3)