import tensorflow as tf

# from tensorflow.contrib.layers import fully_connected
from hiddenbuilder.cnn.ConvHiddenBuilder import ConvHiddenBuilder 
from networkhelpers.Conv import Conv2d


class FFConvHiddenBuilder(ConvHiddenBuilder):
  """
  Feed forward Conv Hidden Builder
  This is used to build a Feed Forward Neural Network
  """

  def build_convolution(self):
    """
    This builds the convolutional part of the network.
    """
    hidden_sizes = self.hidden_sizes.conv_weights
    counter = 0
    name_formatter = "cnn_layer_{}"

    with tf.name_scope(name_formatter.format(counter)):
      with tf.variable_scope("cnn"):
        hidden = Conv2d.conv_layer(
          self.inputs, hidden_sizes[0]
          )
        hidden = Conv2d.max_pool_2x2(hidden)

      for hidden_shape in hidden_sizes[1:]:
        counter += 1

        with tf.variable_scope(name_formatter.format(counter)):
          hidden = Conv2d.conv_layer(hidden, hidden_shape, self.is_training)
          hidden = Conv2d.max_pool_2x2(hidden)

    return hidden