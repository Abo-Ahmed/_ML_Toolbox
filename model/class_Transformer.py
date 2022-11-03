
from os import name

class Transformer(BasicModel): 

        ################
    ## constructor and destructor
    ################
    def __init__(self):
        super().__init__()
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


    def build (self):
        super().build()

        sequence_length = 1
        embed_dim = 3
        dense_dim = 4
        num_heads = 1
        classes = self.nClasses

        inputs = keras.Input(shape=( None , 100 , 100  , 3))
        x = self.PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = self.TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(classes, activation="softmax")(x)
        self.model = keras.Model(inputs, outputs)

        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
