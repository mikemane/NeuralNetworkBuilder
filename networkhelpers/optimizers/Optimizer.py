import tensorflow as tf

from abc import abstractmethod

class Optimizer(object):
  """
  Different Variants of optimizers  
  """

  def __init__(self, loss, learning_rate):
    self.loss = loss
    self.learning_rate = learning_rate
  
  @abstractmethod
  def optimize(self):
    """
    Optimize this values
    """

