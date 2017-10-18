import pickle
import tensorflow as tf
import numpy as np


from collections import namedtuple
from trainer.Config import Config
from trainer.Config import Value, ConvHidden

from networkbuilder.ConvNetworkBuilder import ConvNetworkBuilder
from hiddenbuilder.cnn.FFConvHiddenBuilder import FFConvHiddenBuilder 

from trainer.Trainer import Trainer, TrainerConfig

from dataset.ConvMNIST import ConvMNIST

BATCH_SIZE = 10
LEARNING_RATE = 0.0001
EPOCHS = 100
KEEP_PROB = 0
DISPLAY_STEP = 500

def main():
  """
  Testing the convolutional example on the mnist dataset.
  """

  dataset = ConvMNIST(64)
  print(dataset.get_train().x.shape)


  inputs = Value(type=tf.float32, shape=(None, 28, 28, 1), cls = None)
  targets = Value(type=tf.int64, shape=(None), cls = 10)
  learning_rate = 0.0001

  fc_hidden = [1024, 500]
  c_h = [
    (3, 3, 1, 32),
    (3, 3, 32, 64)
  ]
  conv_hidden = ConvHidden(conv_weights=c_h, fc_weights=fc_hidden)

  config = Config(inputs, targets, conv_hidden, learning_rate)

  network = ConvNetworkBuilder(config)
  hidden = FFConvHiddenBuilder()
  _ = network.build_network(hidden)


  train_config = TrainerConfig(
        epochs = EPOCHS, display_after = DISPLAY_STEP, 
        keep_prob = KEEP_PROB,checkpoint_path=None, 
        summary_path=None
        )

  trainer = Trainer(network, train_config)
  trainer.train(dataset)



if __name__ == '__main__':
  main()