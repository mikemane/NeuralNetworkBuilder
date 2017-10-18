import os, sys
import tensorflow as tf

from abc import abstractmethod


class Builder(object):
  """
    Neural Network Builder.
    Input --> Some Network --> loss(logits, targets) --> optimise(loss).
  """

  def __init__(self, config):
    self.config = config
    self._inputs = None
    self._targets = None
    self._is_training = None
    self._keep_prob = None
    self.optimiser = None
    self.loss = None
    self.accuracy = None

  def build_network(self, hidden_layer_builder):
    """
      Builds the network.
    """
    _, targets = self.initialise_input()
    # Should have passed the inputs and the config to the strategy
    logits = self.create_network(hidden_layer_builder)
    self.loss = self.compute_loss(logits, targets)
    self.optimiser = self.optimise(self.loss)
    self.accuracy = self.calculate_accuracy(logits, targets)
    return self.loss, self.optimiser, self.accuracy

  @property
  def inputs(self):
    """
      Returns the inputs of the network.
    """
    if self._inputs == None:
      raise NotImplementedError("Input Hasnt been set")
    return self._inputs

  @property
  def targets(self):
    """
      Returns the targets of the network.
    """
    if self._targets == None:
      raise NotImplementedError("Targets Hasnt been set")
    return self._targets

  @property
  def keep_prob(self):
    """
    Keep Probability
    """
    return self._keep_prob

  @property
  def is_training(self):
    """
    Is training attributes, for batch norm, dropout and so forth.
    """
    if self._is_training == None:
      self._is_training = tf.placeholder(tf.bool, shape=())
    return self._is_training

  def initialise_input(self):
    """
      This initialise this inputs based on the specification of the config file.
    """
    self._inputs = tf.placeholder(
      self.config.input.type , self.config.input.shape, name="inputs"
      )
    self._targets = tf.placeholder(
      self.config.target.type, self.config.target.shape, name = "targets"
      )
    self._is_training = tf.placeholder(tf.bool, shape=())
    # For dropout probabilities
    self._keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")
    return self._inputs, self._targets

  def create_network(self, hidden_builder):
    """
    Create a network based on the specified information.
    """
    return hidden_builder(
        self.inputs, self.config.hidden_sizes, 
        self.config.target.cls, 
        self.keep_prob, self.is_training
    )

  def compute_loss(self, logits, targets):
    """
    This will compute loss on the logits and targets.
    """

    raise NotImplementedError("Should create the loss based on teh logits and targets.")

  def optimise(self, loss):
    """
    This will optimise the loss.
    """
    return tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)

  def calculate_accuracy(self, logits, targets):
    """
    Get the performance of the network.
    """
    raise NotImplementedError("Have not implemented accuracy metric")

