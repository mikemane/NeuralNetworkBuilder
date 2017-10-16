import tensorflow as tf

from networkbuilder.Builder import Builder

class ConvNetworkBuilder(Builder):
  """
  Convolutional builder
  """

  # def create_network(self, strategy):
  #   # convolutional_strategy = ConvolutionalStrategy()

  def compute_loss(self, logits, targets):
    # loss = tf.nn.softmax_cross_entropy_with_logits(
    #   labels = targets,
    #   logits = logits
    # )
    # return tf.reduce_mean(loss)
    with tf.name_scope("loss"):
      xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits)
      loss = tf.reduce_mean(xentropy, name="loss")
      return loss


  def calculate_accuracy(self, logits, targets):
    with tf.name_scope("conv_accuracy"):
      correct = tf.nn.in_top_k(logits, targets, 1)
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy