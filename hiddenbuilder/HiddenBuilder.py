from abc import abstractmethod

class HiddenBuilder(object):
  """
  Uses a series of hidden strategy to construct a neural network.
  """

  def __init__(self):
    self._inputs = None 
    self._hidden_sizes = None
    self._keep_prob = None
    self._is_training = None
    self._target_dim = None
  
  def __call__(self, inputs , hidden_sizes, target_dim,  keep_prob, is_training):
    """
    Setting the values
    """
    self._inputs = inputs
    self._hidden_sizes = hidden_sizes 
    self._keep_prob = keep_prob
    self._target_dim = target_dim
    self._is_training = is_training
    return self.build()

  @abstractmethod
  def build(self):
    """
    Build a network according to some specification.
    """

  @property
  def inputs(self):
    """
    Inputs values
    """
    return validate_value(self._inputs, "Invalid Input Values")

  @property
  def hidden_sizes(self):
    """
    Keep Probability
    """
    return validate_value(self._hidden_sizes, "Invalid Hidden Sizes")

  @property
  def keep_prob(self):
    """
    Keep Probability
    """
    return validate_value(self._keep_prob, "Invalid Keep Probability")

  @property
  def target_dim(self):
    """
    Target dimension.
    """
    return validate_value(self._target_dim, "Invalid Target Values")

  @property
  def is_training(self):
    """
    Is training values.
    """
    return validate_value(self._is_training, "Invalid Input v")

def validate_value(value, error_message):
    """
    Validates a value
    """
    if value is None:
      raise TypeError(error_message)
    return value
