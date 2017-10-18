import tensorflow as tf

from abc import abstractmethod
from hiddenbuilder.HiddenBuilder import HiddenBuilder
from networkhelpers.rnn.LSTMCell import LSTMCell



class RNNBuilder(HiddenBuilder):
  """
  RNN Builder for building Recurrent Networks.
  """


  def build(self):
    outputs, states = self.build_rnn_layer()
    logits = self.build_fully_connected_layer(outputs, states)
    return logits
  

  @abstractmethod
  def build_rnn_layer(self):
    """
    Builds the RNN part of the netowrk
    """

  @abstractmethod
  def build_fully_connected_layer(self, rnn_outputs, rnn_states):
    """
    Builds the fully connected part of the RNN netowrk.
    """

  def make_cell(self, hidden_size):
    """
    Creates an RNN Cell based on the specified input
    """
    # Might want to add some orthogonal initializer.
    return LSTMCell(hidden_size)

  
