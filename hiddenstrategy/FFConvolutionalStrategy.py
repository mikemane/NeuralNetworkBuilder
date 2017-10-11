import tensorflow as tf

# from tensorflow.contrib.layers import fully_connected
from hiddenstrategy.HiddenStrategy import HiddenStrategy
from networkhelpers.Conv import Conv2d


class ConvolutionalStrategy(HiddenStrategy):

  def build(self):
    """
    Builds the convolutional neural network.
    input --> conv-layer --> fc --> logits.
    """
    # Builds the convolutional layers
    # Builds the fully connected layers


    hidden_sizes = self.network.config.hidden_sizes.conv
    fc_sizes = self.network.config.hidden_sizes.fc

    with tf.name_scope("cnn"):
      hidden = Conv2d.conv_layer(
        self.network.inputs, hidden_sizes[0])
      hidden = Conv2d.max_pool_2x2(hidden)
      for hidden_shape in hidden_sizes[1:]:
        hidden = Conv2d.conv_layer(hidden, hidden_shape)
        hidden = Conv2d.max_pool_2x2(hidden)
      
      # TODO: Find a way to use the feed forward builder to build this network.
      hidden_to_fc = self.network.config.input.shape[1] // (len(hidden_sizes) * 2)
      last_bias = hidden_sizes[-1][3]

      fc = tf.reshape(hidden, [-1, hidden_to_fc * hidden_to_fc * last_bias])

      for shape in fc_sizes:
        fc = Conv2d.full_layer(fc, shape)
        fc = tf.nn.relu(fc)
        # TODO: Implement the dropout layer in a way.
        # fc = tf.nn.droput(fc, keep_prob=0.4)

      logits = Conv2d.full_layer(
        fc, self.network.config.target.cls
        )
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
      """
      raise NotImplementedError("Should Be implemented")
    
    def build_fully_connected(self, conv_part):
      """
      this builds the fully connected part.
      """
      raise NotImplementedError("Should Be implemented")
