from collections import namedtuple

class Config(object):

  def __init__(
    self, 
    input,
    target,
    hidden_sizes, 
    learning_rate
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
    # if not isinstance(self._input, tuple):
    #   raise TypeError("Input shape type must be a tuple instead got {}".format(type(self._input))
    return self._input

  @property
  def target(self):
    """
    Returns the target size.
    """
    # if not isinstance(self._target, tuple):
    #   raise TypeError("Target type must be tuple")
    return self._target

  @property
  def hidden_sizes(self):
    """
    Returns the hidden sizes.
    """
    if not isinstance(self._hidden_sizes, list):
      raise TypeError("Mismatched types")
    return self._hidden_sizes
  
Value = namedtuple("Value", ["type", "shape", "cls"])