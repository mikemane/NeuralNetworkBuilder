import tensorflow as tf

from dataset.Dataset import Dataset, DataValues
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data")

N_STEPS = 28
N_INPUTS = 28

class RNNMNIST(Dataset):
  """
  MNIST DATASET
  """
  def __init__(self, batch_size):
    Dataset.__init__(self, batch_size)

    in_reshape = (-1, N_STEPS, N_STEPS)

    self.set_train(
      DataValues(
        x=mnist.train.images.reshape(*in_reshape), y=mnist.train.labels
        )
      )
    print("Label size is ", self.get_train().y.shape)
    self.set_val(
      DataValues(
        x=mnist.test.images.reshape(*in_reshape), y=mnist.test.labels
        )
      )
    self.set_test(
      DataValues(
        x=mnist.test.images.reshape(*in_reshape), y=mnist.test.labels
        )
     )