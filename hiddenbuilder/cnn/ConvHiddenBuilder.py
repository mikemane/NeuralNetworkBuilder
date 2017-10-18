import tensorflow as tf
import numpy as np

from hiddenbuilder.HiddenBuilder import HiddenBuilder
from networkhelpers.Conv import Conv2d

WITH_MAX_POOL = 2

class ConvHiddenBuilder(HiddenBuilder):
  """
  Builds a Convolutional Layer by Using the Convolutional Part and the Fully Connected Layer.
  """
  def build(self):
    """
    Builds the convolutional neural network.
    input --> conv-layer --> fc --> logits.
    """
    conv_part = self.build_convolution()
    logits = self.build_fully_connected(conv_part)
    return logits


  def is_even(self, number):
    """
    Checks if the number is even.
    """
    result = False
    if (number % 2) == 0:
      result = True
    return result

  def build_convolution(self):
    """
    This builds the convolutional part of the network.
    input --> conv 
    """
    raise NotImplementedError("Build Convolution Should Be implemented")
  
  # def build_fully_connected(self, conv_part):
  #   """
  #   this builds the fully connected part.
  #   """
  #   raise NotImplementedError("Should Be implemented")
  def build_fully_connected(self, conv_part):
    """
    this builds the fully connected part.
    conv ---> fc --> logits
    """
    with tf.variable_scope("cnn_fc"):
      fc_sizes = self.hidden_sizes.fc_weights
      out_dim = conv_part.shape[1] * conv_part.shape[2] * conv_part.shape[3]
      fc_layer = tf.reshape(conv_part, np.asarray([-1, out_dim]))

      for index, shape in enumerate(fc_sizes):
        fc_layer = Conv2d.full_layer(fc_layer, shape)
        fc_layer = tf.nn.relu(fc_layer)
        if index < len(fc_sizes) - 2:
          fc_layer = tf.nn.dropout(fc_layer, keep_prob=self.keep_prob)

      logits = Conv2d.full_layer(
        fc_layer, self.target_dim
        )
    return logits
