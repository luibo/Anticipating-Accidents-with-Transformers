import numpy as np
import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, num_features, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length * num_features, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        inputs = tf.cast(inputs, self.compute_dtype)
        length = tf.shape(inputs)[1]
        #positions = np.arange(start=0, stop=length * self.num_features, step=1)
        positions = tf.range(start=0, limit=length * self.num_features, delta=1)
        embedded_positions = self.position_embeddings(positions)
        embedded_positions = tf.reshape(embedded_positions, (-1, self.sequence_length, self.num_features, self.output_dim))
        return inputs + embedded_positions
