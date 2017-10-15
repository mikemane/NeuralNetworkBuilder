import pickle
import tensorflow as tf
import numpy as np


from collections import namedtuple
from trainer.Config import Config
from trainer.Config import Value, FCHidden

from networkbuilder.FFNetworkBuilder import FFNetworkBuilder 
from hiddenbuilder.fc.FFHiddenBuilder import FFHiddenBuilder

from trainer.Trainer import Trainer

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")


def main():
  """
    Testing the feedforward framework on the mnist dataset.
  """
  inputs = Value(type=tf.float32, shape=(None, 784), cls=None)
  targets = Value(type=tf.int64, shape=(None), cls=10)
  fc_hidden = FCHidden(weights=[300, 150])
  learning_rate = 0.0001

  config = Config(inputs, targets, fc_hidden, learning_rate)

  network_builder = FFNetworkBuilder(config)
  hidden_builder = FFHiddenBuilder(network_builder)
  loss, optimiser, accuracy = network_builder.build_network(hidden_builder)

  batch_size = 64
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for i in range(200):
      for _ in range(mnist.train.num_examples//batch_size):
        train, labels = mnist.train.next_batch(batch_size)
        train = train.reshape((batch_size, inputs.shape[1]))

        _, l = sess.run([optimiser, loss], feed_dict={
          network_builder.inputs: train,
          network_builder.targets: labels,
          network_builder.is_training: True
        })

        counter += 1
        if (counter % 100) == 0:
          test_images = mnist.test.images
          test_labels = mnist.test.labels

          feed_dict = {
            network_builder.inputs: test_images,
            network_builder.targets: test_labels,
            network_builder.is_training: False
          }
          acc = sess.run([
            accuracy
            ], feed_dict=feed_dict)
          print(
            "Epoch {}/{}, Train loss {}, Test Accuracy {}".format(
            i ,200 ,l , acc)
            )


if __name__ == '__main__':
  main()