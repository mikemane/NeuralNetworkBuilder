import sys, os
import pickle
import tensorflow as tf


from collections import namedtuple
from trainer.Config import Config
from trainer.Config import Value, FCHidden

from networkbuilder.FFNetworkBuilder import FFNetworkBuilder 
from hiddenbuilder.fc.FFHiddenBuilder import FFHiddenBuilder

from trainer.Trainer import Trainer, TrainerConfig
from dataset.MNIST import MNIST

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data")

BATCH_SIZE = 10

def main():
  """
    Testing the feedforward framework on the mnist dataset.
  """
  dataset = MNIST(BATCH_SIZE)
  
  inputs = Value(type=tf.float32, shape=(None, 784), cls=None)
  targets = Value(type=tf.int64, shape=(None), cls=10)
  fc_hidden = FCHidden(weights=[300, 150])
  learning_rate = 0.0001

  config = Config(inputs, targets, fc_hidden, learning_rate)

  network_builder = FFNetworkBuilder(config)
  hidden_builder = FFHiddenBuilder(network_builder)
  _ = network_builder.build_network(hidden_builder)

  train_config = TrainerConfig(
          epochs = 100, display_after = 500, 
          keep_prob = 0.5,checkpoint_path=None, 
          summary_path=None
          )
  trainer = Trainer(network_builder, train_config)
  trainer.train(dataset)


if __name__ == '__main__':
    main()
