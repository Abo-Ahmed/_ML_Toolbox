
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
            
            print(">>> PositionalEmbedding 0")
            self.position_embeddings = layers.Embedding(
                input_dim=sequence_length, output_dim=output_dim
            )
            print(">>> PositionalEmbedding 1")
            self.sequence_length = sequence_length
            print(">>> PositionalEmbedding 2")
            self.output_dim = output_dim
            print(">>> PositionalEmbedding 3")

        def call(self, inputs):
            # The inputs are of shape: `(batch_size, frames, num_features)`
            length = tf.shape(inputs)[1]
            print(">>> PositionalEmbedding 4")
            positions = tf.range(start=0, limit=length, delta=1)
            print(">>> PositionalEmbedding 5")
            embedded_positions = self.position_embeddings(positions)
            print(">>> PositionalEmbedding 6")
            return inputs + embedded_positions

        def compute_mask(self, inputs, mask=None):
            print(">>> PositionalEmbedding 7")
            mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
            print(">>> PositionalEmbedding 8")
            return mask

    class TransformerEncoder(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
            super().__init__(**kwargs)
            
            print(">>> TransformerEncoder 0")
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim, dropout=0.3
            )
            print(">>> TransformerEncoder 1")
            self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
            )
            print(">>> TransformerEncoder 2")
            self.layernorm_1 = layers.LayerNormalization()
            print(">>> TransformerEncoder 3")
            self.layernorm_2 = layers.LayerNormalization()
            print(">>> TransformerEncoder 4")

        def call(self, inputs, mask=None):
            if mask is not None:
                mask = mask[:, tf.newaxis, :]

            print(">>> TransformerEncoder 5")
            print(inputs)
            print(mask)
            attention_output = self.attention(inputs, inputs, attention_mask=mask)
            print(">>> TransformerEncoder 6")
            proj_input = self.layernorm_1(inputs + attention_output)
            print(">>> TransformerEncoder 7")
            proj_output = self.dense_proj(proj_input)
            print(">>> TransformerEncoder 8")
            temp = proj_input + proj_output
            print(">>> TransformerEncoder 9")
            return self.layernorm_2(temp)

    def build (self):
        super().build()

        sequence_length = 1
        embed_dim = 1
        dense_dim = 1
        num_heads = 1
        classes = self.nClasses

        if(handler.isColored):
            depth = 3
        else:
            depth = 1 

        inputs = keras.Input(shape=( handler.imageWidth , handler.imageHeight , depth))
        print(">>> check point 0" , inputs)
        x = inputs
        # x = self.PositionalEmbedding(
        #     sequence_length, embed_dim, name="frame_position_embedding"
        # )(x)
        print(">>> check point 1")
        x = self.TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
        print(">>> check point 2")
        x = layers.GlobalMaxPooling2D()(x)
        print(">>> check point 3")
        x = layers.Dropout(0.5)(x)
        print(">>> check point 4")
        outputs = layers.Dense(classes, activation="softmax")(x)
        outputs = x

        print(">>> check point 5")
        self.model = keras.Model(inputs, outputs)
        print(">>> check point 6")

        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        print(">>> check point 7")
