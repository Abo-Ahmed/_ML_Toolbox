
from os import name

class Transformer(BasicModel): 

        ################
    ## constructor and destructor
    ################
    def __init__(self):
        super().__init__()
        self.MAX_SEQ_LENGTH = 1
        self.NUM_FEATURES = 1024
        self.IMG_SIZE = 128
        self.EPOCHS = 30

        self.batchStart = 0
        self.batchEnd = 200
        self.imageWidth = 128
        self.imageHeight = 128

        self.base_path = '/content/drive/MyDrive/eng-mahmoud/dataSet/danbooru2019/'
        self.images_path = base_path + 'images/'

        self.s = "safe"
        self.e = "explicit"
        self.q = "questionable"

        self.image_info = ['image_name', 'image_rate']
        self.image_dict = []
        self.rates = ["safe" , "questionable" , "explicit"]

        self.test_images = []
        self.test_array = []
        self.train_images = []
        self.train_array = []


    class PositionalEmbedding(layers.Layer):
        def __init__(self, sequence_length, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.position_embeddings = layers.Embedding(
                input_dim=sequence_length, output_dim=output_dim
            )
            self.sequence_length = sequence_length
            self.output_dim = output_dim

        def call(self, inputs):
            # The inputs are of shape: `(batch_size, frames, num_features)`
            length = tf.shape(inputs)[1]
            positions = tf.range(start=0, limit=length, delta=1)
            embedded_positions = self.position_embeddings(positions)
            return inputs + embedded_positions

        def compute_mask(self, inputs, mask=None):
            mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
            return mask



    class TransformerEncoder(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.dense_dim = dense_dim
            self.num_heads = num_heads
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim, dropout=0.3
            )
            self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
            )
            self.layernorm_1 = layers.LayerNormalization()
            self.layernorm_2 = layers.LayerNormalization()

        def call(self, inputs, mask=None):
            if mask is not None:
                mask = mask[:, tf.newaxis, :]

            attention_output = self.attention(inputs, inputs, attention_mask=mask)
            proj_input = self.layernorm_1(inputs + attention_output)
            proj_output = self.dense_proj(proj_input)
            return self.layernorm_2(proj_input + proj_output)

    def get_compiled_model():
        sequence_length = MAX_SEQ_LENGTH
        embed_dim = NUM_FEATURES
        dense_dim = 4
        num_heads = 1
        classes = len(label_processor.get_vocabulary())

        inputs = keras.Input(shape=(None, None))
        x = PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def run_experiment():
        filepath = "/content/video_classifier"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, save_weights_only=True, save_best_only=True, verbose=1
        )

        model = get_compiled_model()
        history = model.fit(
            train_array,
            np.array(train_labeling),
            validation_split=0.15,
            epochs=10
            # callbacks=[checkpoint]
        )

        # model.load_weights(filepath)
        _, accuracy = model.evaluate(test_array, np.array(test_labeling))
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        return model

    def build (self):
        super().build()
        self.model = self.get_compiled_model()
        # self.model = tf.keras.models.Sequential()
        # inputs = tf.keras.Input(shape=(handler.imageWidth, handler.imageHeight, 3), name="input_1")
        # t = keras_layers.Flatten(name="flatten_2") (inputs)
        # t = keras_layers.Dense(handler.imageWidth, activation=tf.nn.relu , name="dense_3") (t)
        # t = keras_layers.Dropout(0.2,name="dropout_4") (t)
        # outputs = keras_layers.Dense(10, activation=tf.nn.softmax,name="dense_5") (t)
        
        # self.model = tf.keras.Model(inputs , outputs)
        # self.model.compile(optimizer='adam',
        #                 loss='sparse_categorical_crossentropy',
        #                 metrics=['accuracy'])


