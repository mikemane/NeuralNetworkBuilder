from collections import namedtuple

class Config(object):

  def __init__(
    self, 
    input,
    target,
    hidden_sizes, 
    learning_rate,
    ):
    self._learning_rate = learning_rate
    self._input = input
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

 
Value = namedtuple("Value", ["type", "shape", "cls"])
FCHidden = namedtuple("FCHidden", ["fc"])
ConvHidden = namedtuple("ConvHidden", ["conv", "fc"])