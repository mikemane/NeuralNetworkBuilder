import tensorflow as tf

from collections import namedtuple

VAL_TRAIN_KEEP_PROB = 0.0
IS_TRAINING = True

# Trainer Config: This represents the conguration for the trianer file.
TrainerConfig = namedtuple(
  "TrainerConfig", [
    "epochs", "display_after", "keep_prob", "checkpoint_path", "summary_path"
    ]
  )

class Trainer(object):

  def __init__(self, network, config):
    print("Initialising Training Set")
    self.network = network
    self.config = config
    self.counter = 0

  def add_summaries(self, writer, counter, *args):
    """
    Adds some summaries.
    """
    for item in args:
      writer.add_summary(item, counter)
  
  def check_validation_set(self,sess, val_set):
    """
    Checks how well data is doing based on the val set.
    """
    val_feed_dict = self.construct_feed_dict(
      val_set.x, val_set.y, VAL_TRAIN_KEEP_PROB, not (IS_TRAINING)
      )
    val_accuracy = sess.run(
      self.network.accuracy, feed_dict = val_feed_dict
      )
    print("Validation Accuracy: {}".format(val_accuracy))
  
  def check_train_set(self, sess, *args):
    """
    Train values
    """
    train_operations = [self.network.optimiser, self.network.loss]
    feed_dict = self.construct_feed_dict(*args)
    return sess.run(train_operations, feed_dict)


  def construct_feed_dict(self, x, y, keep_prob, is_training):
    feed_dict = dict()
    feed_dict[self.network.inputs] = x
    feed_dict[self.network.targets] = y
    feed_dict[self.network.keep_prob] = keep_prob
    feed_dict[self.network.is_training] = is_training
    return feed_dict



  def train(self, dataset):
    """
      Perform training of a network.
    """
    saver = None
    print("About to train Dataset")
    print("-" * 50, "\n")
    with tf.Session() as sess:
      if self.config.checkpoint_path != None:
        saver = tf.train.Saver()
        sess.restore(sess, self.config.checkpoint_path)
      else:
        sess.run(tf.global_variables_initializer())

      display_after = self.config.display_after if self.config.display_after != None else 100

      if self.config.summary_path != None:
        summary_writer = tf.summary.FileWriter(
          self.config.summary_path, sess.graph
          )

      for epoch in range(self.config.epochs):
        for index, (train_x, train_y) in enumerate(
          dataset.next_batch(dataset.get_train())
          ):

          keep_prob = self.config.keep_prob if self.config.keep_prob is not None else 1.0
          train_params = [train_x, train_y, keep_prob, IS_TRAINING]
          optimiser, loss = self.check_train_set(sess, *train_params)

          self.counter += 1
          if (self.counter % display_after) == 0:
            print(
              "Epoch %d/%d, Training loss = %.4f" % (
                epoch, self.config.epochs, loss
                )
              )
            if dataset.get_val() != None:
              self.check_validation_set(sess, dataset.get_val())