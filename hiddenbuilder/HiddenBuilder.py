
class HiddenBuilder(object):
  """
  Uses a series of hidden strategy to construct a neural network.
  """

  def __init__(self, network):
    self.network = network

  def build(self):
    """
    Build a network according to some specification.
    """
    raise NotImplementedError("Implement this in a sub class")


  