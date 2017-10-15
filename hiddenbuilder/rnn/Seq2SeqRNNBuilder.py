from hiddenbuilder.RNNBuilder import RNNBuilder 
from abc import abstractmethod


class Seq2SeqBuilder(RNNBuilder):
  """
  builds the Seq2Seq Module for the RNN network.
  """


  def build(self):
    """
    in --> encoder --> attention --> decoder --> logits
    """
    encoder_output, encoder_states = self.build_encoder()
    attention_output = self.use_attention(
      encoder_output, encoder_states
      )
    decoder_output, decoder_state = self.build_decoder(attention_output)
    logits = self.build_fully_connected_layer(
      decoder_output, decoder_state
      )
    return logits

  @abstractmethod
  def build_encoder(self):
    """
    Builds the encoder part of the seq 2 seq
    in --> encoder
    """

  @abstractmethod
  def use_attention(self, encoder_outputs, encoder_states):
    """
    Builds the attention module of the neuralnetwork.
    encoder --> attention
    """

  @abstractmethod
  def build_decoder(self, attention_out):
    """
    Builds the decoder part of the network
    attention --> decoder 
    """