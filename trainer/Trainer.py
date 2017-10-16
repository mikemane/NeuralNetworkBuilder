import tensorflow as tf

from collections import namedtuple

VAL_TRAIN_KEEP_PROB = 0.0

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
    # print("Test Validation set here")
    val_x = val_set.x
    val_y = val_set.y

    val_feed_dict = {
      self.network.inputs: val_x,
      self.network.targets: val_y,
      self.network.keep_prob: VAL_TRAIN_KEEP_PROB,
      self.network.is_training: False
    }
    val_accuracy = sess.run(
      self.network.accuracy, feed_dict = val_feed_dict
      )
    print("Validation Accuracy: {}".format(val_accuracy))


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
          feed_dict = {
            self.network.inputs: train_x,
            self.network.targets: train_y,
            self.network.is_training: True
          }

          if self.config.keep_prob != None:
            feed_dict[self.network.keep_prob] = self.config.keep_prob

          _, loss = sess.run(
              [
                self.network.optimiser,
                self.network.loss
              ],
              feed_dict = feed_dict
              )

          self.counter += 1
          if (self.counter % display_after) == 0:
            print(
              "Epoch %d/%d, Training loss = %.4f" % (
                epoch, self.config.epochs, loss
                )
              )
            if dataset.get_val() != None:
              self.check_validation_set(sess, dataset.get_val())


