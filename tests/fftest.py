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


BATCH_SIZE = 10
LEARNING_RATE = 0.0001
EPOCHS = 100
KEEP_PROB = 0.05
DISPLAY_STEP = 500

def main():
  """
    Testing the feedforward framework on the mnist dataset.
  """
  dataset = MNIST(BATCH_SIZE)
  
  inputs = Value(type=tf.float32, shape=(None, 784), cls=None)
  targets = Value(type=tf.int64, shape=(None), cls=10)
  fc_hidden = FCHidden(weights=[300, 150])

  config = Config(inputs, targets, fc_hidden, LEARNING_RATE)

  network_builder = FFNetworkBuilder(config)
  hidden_builder = FFHiddenBuilder()
  _ = network_builder.build_network(hidden_builder)

  train_config = TrainerConfig(
          epochs = EPOCHS, display_after = DISPLAY_STEP, 
          keep_prob = KEEP_PROB,checkpoint_path=None, 
          summary_path=None
          )
  trainer = Trainer(network_builder, train_config)
  trainer.train(dataset)


if __name__ == '__main__':
    main()
