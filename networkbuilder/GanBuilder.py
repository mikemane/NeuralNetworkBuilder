import tensorflow as tf
import numpy as np

from builder import Builder 


class GanBuilder(Builder):
  """
  Builds a gan network 
  """

  def create_network(self, inputs):
    pass
    


  def compute_loss(self, logits, targets):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
       labels = targets, logits = logits))
    return loss

  def optimiser(self, loss, learning_rate):
    return tf.train.AdamOptimiser(self.config.learning_rate).minimize(loss)



