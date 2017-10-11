import pickle
import tensorflow as tf
import numpy as np


from collections import namedtuple
from trainer.Config import Config
from trainer.Config import Value, FCHidden, ConvHidden

from networkbuilder.FeedForwardBuilder import FeedForwardBuilder 
from networkbuilder.ConvolutionalBuilder import ConvolutionalBuilder
from hiddenstrategy.FeedForwardStrategy import FeedForwardStrategy
from hiddenstrategy.ConvolutionalStrategy import  ConvolutionalStrategy

from trainer.Trainer import Trainer

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")

# FILENAME = "data.pkl"



def main():
  # print(mnist.train.images.shape)
  # with open(FILENAME, "rb") as f:
  #   data = pickle.load(f)
  #   print(data.keys())

  #   for i in range(5):
  #     print(data["Xs"][i])
  #     print(data["Ys"][i])
  # inputs = Value(type=tf.float32, shape=(None, 784), cls=None)
  # targets = Value(type=tf.int64, shape=(None), cls=10)


  # fc_hidden = FCHidden(fc=[300, 150])
  # learning_rate = 0.0001

  # config = Config(inputs, targets, fc_hidden, learning_rate)

  # feed_foward_network = FeedForwardBuilder(config)
  # feed_foward_strategy = FeedForwardStrategy(feed_foward_network)
  # loss, optimiser, accuracy = feed_foward_network.build_network(feed_foward_strategy)

  inputs = Value(type=tf.float32, shape=(None, 28, 28, 1), cls = None)
  targets = Value(type=tf.int64, shape=(None), cls = 10)
  learning_rate = 0.0001

  fc_hidden = [1024, 500]
  c_h = [
    (5, 5, 1, 32),
    (5, 5, 32, 64)
  ]
  conv_hidden = ConvHidden(conv=c_h, fc=fc_hidden)

  config = Config(inputs, targets, conv_hidden, learning_rate)

  network = ConvolutionalBuilder(config)
  conv2dstrategy = ConvolutionalStrategy(network)
  loss, optimiser, accuracy = network.build_network(conv2dstrategy)

  batch_size = 64
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for i in range(200):
      for _ in range(mnist.train.num_examples//batch_size):
        train, labels = mnist.train.next_batch(batch_size)
        # train = train.reshape((batch_size, inputs.shape[1]))
        train = train.reshape(-1, 28, 28, 1)

        _, l = sess.run([optimiser, loss], feed_dict={
          network.inputs: train,
          network.targets: labels,
          network.is_training: True
        })

        counter += 1
        if (counter % 100) == 0:
          test_train = mnist.test.images.reshape(-1, 28,28, 1)
          test_labels = mnist.test.labels

          feed_dict = {
            network.inputs: test_train,
            network.targets: test_labels,
            network.is_training: False
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