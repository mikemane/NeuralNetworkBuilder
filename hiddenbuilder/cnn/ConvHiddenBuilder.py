import tensorflow as tf

# from tensorflow.contrib.layers import fully_connected
from hiddenbuilder.HiddenBuilder import HiddenBuilder
from networkhelpers.Conv import Conv2d


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
      hidden_sizes = self.network.config.hidden_sizes.conv_weights
      fc_sizes = self.network.config.hidden_sizes.fc_weights

      hidden_to_fc = self.network.config.input.shape[1] // (len(hidden_sizes) * 2)
      last_bias = hidden_sizes[-1][3]

      fc_layer = tf.reshape(conv_part, [-1, hidden_to_fc * hidden_to_fc * last_bias])

      for index, shape in enumerate(fc_sizes):
        fc_layer = Conv2d.full_layer(fc_layer, shape)


        fc_layer = tf.nn.relu(fc_layer)
        if index < len(fc_sizes) - 2:
          fc_layer = tf.nn.dropout(fc_layer, keep_prob=self.network.keep_prob)

      logits = Conv2d.full_layer(
        fc_layer, self.network.config.target.cls
        )
    return logits
