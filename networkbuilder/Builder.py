import os, sys
import tensorflow as tf


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

  def build_network(self):
    """
      Builds the network.
    """
    # Initalse he inputs based on the shape provided.
    inputs, targets = self.initialise_input()
    # Create the network according to the specification
    logits = self.create_network()
    # Compute the losses from the inputs and outputs
    loss = self.compute_loss(logits, targets)
    # Optimises the losses
    optimiser = self.optimise(loss)
    #  takes in input sequence and labels and computes the loss
    # train_values(input, data)
    return loss, optimiser

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
    return self._inputs, self._targets

  def create_network(self):
    """
    Create a network based on the specified information.
    """
    raise NotImplementedError("Create the neto")

  def compute_loss(self, logits, targets):
    """
    This will compute loss on the logits and targets.
    """

    raise NotImplementedError("Should create the loss based on teh logits and targets.")

  def optimise(self, loss):
    """
    This will optimise the loss.
    """
    raise NotImplementedError("This will optimise the loss.")

  def accuracy(self, loss):
    """
    Get the performance of the network.
    """
    raise NotImplementedError("Have not implemented accuracy metric")

