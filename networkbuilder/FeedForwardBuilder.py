import tensorflow as tf

from networkbuilder.Builder import Builder

class FeedForwardBuilder(Builder):
  """
  Deep Feed forward Builder.
  input --> DFFNN --> optimise(mean(loss(logits, targets))
  """
  def compute_loss(self, logits, targets):
    with tf.name_scope("ff_loss"):
      loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits,
          labels=targets
        )
      )
      return loss

  def optimise(self, loss):
    """
    Perform optimisation, gradient clipping, l1 - l2 loss and so forth.
    """
    optimiser = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
    return optimiser

  def accuracy(self, logits, targets):
    with tf.name_scope("eval"):
      correct = tf.nn.in_top_k(logits, targets, 1)
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
