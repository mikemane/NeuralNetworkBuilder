from collections import namedtuple

# Values for the input and the output layers
# @type: The input or output layers datatype
# @shape: This is the shape of the input or the output layer.
# @cls: The output layer size: represents a single integer. 
Value = namedtuple("Value", ["type", "shape", "cls"])

# Used to get the layers of the fully connected layers
# @fc: List of weights for fully connected layer e.g [10, 10 , 4]
FCHidden = namedtuple("FCHidden", ["weights"])

# Blue print of the convolutional layer
# @conv_weights: list of convolutional weights [(filter_w, filter_h, channel, size)]
# @fc_weights: Fully Connected weights [weight] => [100, 50, 2]
ConvHidden = namedtuple("ConvHidden", ["conv_weights", "fc_weights"])

# Might be used for clarity while building convolutional networks.
ConvLayer = namedtuple(
  "ConvLayer", [
    "filter_w", "filter_h", "channels","filter_size", "strides_w", "stride_h"
    ]
  )


class Config(object):
  """
  Configuration file for building deep neural networks.
 """

  def __init__(
    self, 
    invalue,
    target,
    hidden_sizes, 
    learning_rate,
    ):
    self._learning_rate = learning_rate
    self._input = invalue
    self._target = target
    self._hidden_sizes = hidden_sizes

  @property
  def learning_rate(self):
    """
    Returns the learning rate.
    """
    if not isinstance(self._learning_rate, float):
      raise TypeError("Type must be a tuple")
    return self._learning_rate

  @property
  def input(self):
    """
    Returns the input size.
    """
    return self._input

  @property
  def target(self):
    """
    Returns the target size.
    """
    return self._target

  @property
  def hidden_sizes(self):
    """
    Returns the hidden sizes.
    """
    return self._hidden_sizes

 
