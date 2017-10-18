import tensorflow as tf
import numpy as np


from hiddenbuilder.RNNBuilder import RNNBuilder
from networkhelpers.rnn.LSTMCell import LSTMCell

class SimpleRNNBuilder(RNNBuilder):
  """
  Builds an Simple RNN layer from scratch,
  """

  
  def build_rnn_layer(self):
    # In --> RNN ---> 
    with tf.name_scope("rnn"):
      hidden_size = self.network.config.hidden_size.weights
      depth = tf.network.config.hidden_sizes.depth

      hidden = tf.contrib.rnn.MultiRNNCell(
        [self.make_cell(hidden_size) for _ in range(depth)]
        )
      outputs, states = tf.nn.dynamic_rnn(
        hidden, self.network.inputs
        )

      return outputs

  def build_fully_connected_layer(self, rnn_layer):
    # RNNLayer ---> FCLayer --> Logits
    print("Building the fully connectrd part of the RNN layer.")
      
  



      
