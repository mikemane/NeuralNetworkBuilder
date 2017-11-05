import pickle
import tensorflow as tf
import numpy as np

from collections import namedtuple
from trainer.Config import Config
from trainer.Config import Value, RNNHidden 

from networkbuilder.ConvNetworkBuilder import ConvNetworkBuilder 
from hiddenbuilder.rnn.SimpleRNNBuilder import SimpleRNNBuilder 

from trainer.Trainer import Trainer, TrainerConfig
from dataset.RNNMNIST import RNNMNIST


BATCH_SIZE = 10
RNN_HIDDEN = 150
LEARNING_RATE = 0.0001
EPOCHS = 100
KEEP_PROB = 0
DISPLAY_STEP = 500

N_INPUT = 28
N_STEPS = 28

def main():
  """
  Testing the convolutional example on the mnist dataset.
  """

  dataset = RNNMNIST(BATCH_SIZE)
  print(dataset.get_train().y.shape)

  in_shape = (None, N_STEPS, N_INPUT)

  inputs = Value(type=tf.float32, shape=in_shape, cls = None)
  targets = Value(type=tf.int32, shape=(None), cls = 10)

  fc_hidden = [500, 150]
  rnn_config = RNNHidden(
    rnn_weights=RNN_HIDDEN , depth=1, fc_weights=fc_hidden
    )
  config = Config(inputs, targets, rnn_config, LEARNING_RATE)

  network = ConvNetworkBuilder(config)
  hidden = SimpleRNNBuilder()
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