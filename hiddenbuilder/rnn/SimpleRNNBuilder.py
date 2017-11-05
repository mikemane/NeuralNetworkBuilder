import tensorflow as tf
import numpy as np

from hiddenbuilder.rnn.RNNBuilder import RNNBuilder
# from hiddenbuilder.fc.FullyConnectedBuilder import FullyConnectedBuilder
from hiddenbuilder.fc.FullyConnectedBuilder import FullyConnectedBuilder

class SimpleRNNBuilder(RNNBuilder):
  """
  Builds an Simple RNN layer from scratch,
  """
  
  def build_rnn_layer(self):
    """
    Input --> RNNLayer.
    """
    with tf.variable_scope("rnn_layer"):
      with tf.name_scope("rnn"):
        # TODO: Change this strategy to account for both weights and depth
        # print(self.hidden_sizes)
        hidden_size = self.hidden_sizes.rnn_weights
        depth = self.hidden_sizes.depth

        hidden = tf.contrib.rnn.MultiRNNCell(
          [self.make_cell(hidden_size) for _ in range(depth)]
          )
        outputs, states = tf.nn.dynamic_rnn(
          hidden, self.inputs, dtype=tf.float32
          )
        return (outputs, states)

  # def build_fully_connected_layer(self, rnn_layer):
  def build_fully_connected_layer(self, rnn_outputs, rnn_states):
    """
     RNNLayer ----> Fully Connected layer.
    """
    fc_hidden_size = self.hidden_sizes.fc_weights
    fc_layer = FullyConnectedBuilder()
    logits = fc_layer(
        rnn_states[0].h, fc_hidden_size, self.target_dim, self.keep_prob, self.is_training
    )
    return logits