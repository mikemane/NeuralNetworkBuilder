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

  def build_network(self, strategy):
    """
      Builds the network.
    """
    # Initalse he inputs based on the shape provided.
    inputs, targets = self.initialise_input()
    # Create the network according to the specification
    logits = self.create_network(strategy)
    # Compute the losses from the inputs and outputs
    self.loss = self.compute_loss(logits, targets)
    # Optimises the losses
    self.optimiser = self.optimise(self.loss)
    #  takes in input sequence and labels and computes the loss
    # train_values(input, data)
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

  def create_network(self, strategy):
    """
    Create a network based on the specified information.
    """
    return strategy.build()

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

