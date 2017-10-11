import tensorflow as tf

# from tensorflow.contrib.layers import fully_connected
from hiddenstrategy.HiddenStrategy import HiddenStrategy
from networkhelpers.Conv import Conv2d


class ConvolutionalStrategy(HiddenStrategy):
  """
  Builds a Convolutional Layer by Using the Convolutional Part and the Fully Connected Layer.
  """
  def build(self):
    """
    Builds the convolutional neural network.
    input --> conv-layer --> fc --> logits.
    """
    # Builds the convolutional layers
    # Builds the fully connected layers
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
    """
    raise NotImplementedError("Should Be implemented")
  
  def build_fully_connected(self, conv_part):
    """
    this builds the fully connected part.
    """
    raise NotImplementedError("Should Be implemented")
