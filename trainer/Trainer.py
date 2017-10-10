import tensorflow as tf



class Trainer(object):

  def __init__(self, network, epochs, show_after):
    self.epochs = epochs
    self.show_after = show_after
    self.network = network


  def train(self, dataset, checkpoint_path = None):

    with tf.Session() as sess:
      if checkpoint_path:
        tf.train.Saver()
        sess.restore(sess, checkpoint_path)
      else:
        sess.run(tf.global_variables_initializer())

      for epoch in range(self.epochs):
        train_batch, label_batch = dataset.next_batch()
        feed_dict = {
          self.network.inputs: train_batch,
          self.network.targets: label_batch
        }
        sess.run(self.network.optimizer)