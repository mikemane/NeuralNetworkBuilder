import tensorflow as tf

# from tensorflow.contrib.layers import fully_connected
from hiddenbuilder.cnn.ConvolutionalBuilder import ConvolutionalBuilder 
from networkhelpers.Conv import Conv2d


class FFConvolutionalBuilder(ConvolutionalBuilder):
  """
  This is used to build a Feed Forward Neural Network
  """

  def build_convolution(self):
    """
    This builds the convolutional part of the network.
    """
    hidden_sizes = self.network.config.hidden_sizes.conv_weights

    with tf.name_scope("cnn"):
      hidden = Conv2d.conv_layer(
        self.network.inputs, hidden_sizes[0])
      hidden = Conv2d.max_pool_2x2(hidden)
      for hidden_shape in hidden_sizes[1:]:
        hidden = Conv2d.conv_layer(hidden, hidden_shape)
        hidden = Conv2d.max_pool_2x2(hidden)
    return hidden