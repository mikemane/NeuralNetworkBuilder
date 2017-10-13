import tensorflow as tf

from collections import namedtuple



class Trainer(object):

  def __init__(self, network, testconfig):
    self.network = network
    self.config = testconfig
    self.counter = 0


  def train(self, dataset):
    """
      Perform training of a network.
    """
    with tf.Session() as sess:
      if self.config.checkpoint_path != None:
        tf.train.Saver()
        sess.restore(sess, self.config.checkpoint_path)
      else:
        sess.run(tf.global_variables_initializer())

      display_after = self.config.display_after if self.config.display_after != None else 100


      for _ in range(self.config.epochs):
        for train_batch, label_batch in dataset.next_batch():
          feed_dict = {
            self.network.inputs: train_batch,
            self.network.targets: label_batch
          }
          if self.config.keep_prob != None:
            feed_dict[self.network.keep_prob] = self.config.keep_prob
          sess.run(self.network.optimizer, feed_dict = feed_dict)

          self.counter += 1
          if self.counter % display_after == 0:
            print("Display Operation.")
            if dataset.val != None:
              print("Test Validation set here")
              val_train = dataset.val.train
              val_label = dataset.val.labels

              val_feed_dict = {
                self.network.inputs: val_train,
                self.network.targets: val_label,
                self.network.keep_prob: 0.0,
                self.network.is_training: False
              }

              val_accuracy = sess.run(
                self.network.accuracy, feed_dict = val_feed_dict
                )

              print("Validation Accuracy: {}".format(val_accuracy))




        

TestConfig = namedtuple(
  "TestConfig", ["epochs", "display_after", "keep_prob", "checkpoint_path"]
  )