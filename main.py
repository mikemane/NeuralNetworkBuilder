import pickle
import tensorflow as tf
import numpy as np


from collections import namedtuple
from trainer.Config import Config
from trainer.Config import Value 
from networkbuilder.FeedForwardBuilder import FeedForwardBuilder
from trainer.Trainer import Trainer

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")

# FILENAME = "data.pkl"



def main():
  # with open(FILENAME, "rb") as f:
  #   data = pickle.load(f)
  #   print(data.keys())

  #   for i in range(5):
  #     print(data["Xs"][i])
  #     print(data["Ys"][i])
  inputs = Value(type=tf.float32, shape=(None, 784), cls=None)
  targets = Value(type=tf.int64, shape=(None), cls=10)

  hidden_size = [300, 150]
  learning_rate = 0.0001
  config = Config(inputs, targets, hidden_size, learning_rate)

  feed_foward_network = FeedForwardBuilder(config)
  loss, optimiser = feed_foward_network.build_network()

  batch_size = 64
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for i in range(200):
      for _ in range(mnist.train.num_examples//batch_size):
        train, labels = mnist.train.next_batch(batch_size)
        train = train.reshape((batch_size, inputs.shape[1]))

        _, l = sess.run([optimiser, loss], feed_dict={
          feed_foward_network.inputs: train,
          feed_foward_network.targets: labels,
          feed_foward_network.is_training: True
        })

        counter += 1
        if (counter % 100) == 0:
          print("Epoch {}/{}, Loss {}".format(i,200, l))


if __name__ == '__main__':
  main()