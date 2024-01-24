import numpy as np
import tensorflow as tf
import keras

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

  
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
  

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    self.d_model = d_model
    self.warmup_steps = warmup_steps

  def call(self, step):
    step = tf.cast(step, dftype=tf.float32)
    d_model_rsqrt = tf.math.rsqrt(self.d_model)
    step_rsqrt = tf.math.rsqrt(step)
    ws_pow = self.warmup_steps ** -1.5

    return d_model_rsqrt * tf.math.minimum(step_rsqrt, step * ws_pow)



def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)
