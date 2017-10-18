import tensorflow as tf

from networkhelpers.optimizers.optimizer import Optimizer

class ExpOptimizer(Optimizer):

  def __init__(self, loss, learning_rate, clip_threshold = 5.):
    Optimizer.__init__(self, loss, learning_rate)
    self.clip_threshold = clip_threshold

  def optimize(self): 
    with tf.name_scope("optimizer"):
      """
      Optimize for exploding gradients.
      """
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      grads = optimizer.compute_gradient(self.loss)
      capped_gradients = [
        (
          tf.clip_by_value(
            grad, -self.clip_threshold, self.clip_threshold)
            )
            for grad in grads
        ]
      train_op = optimizer.apply_gradient(capped_gradients)
      return train_op

  def clip_exploding_gradients(self):
    """
    Use for clipping exploding gradients.
    """
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    # grads = optimizer.compute_gradient(self.loss)
    trainable_variables = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_variables)
    clipped_gradients = tf.clip_by_global_norm(
      gradients, 
      self.clip_threshold
    )
    train_op = optimizer.apply_gradients(
      zip(clipped_gradients, trainable_variables)
    )
    return train_op