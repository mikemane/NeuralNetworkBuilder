import tensorflow as tf

from dataset.Dataset import Dataset, DataValues
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data")

class MNIST(Dataset):
  """
  MNIST DATASET
  """
  def __init__(self, batch_size):
    Dataset.__init__(self, batch_size)
    self.set_train(DataValues(x=mnist.train.images, y=mnist.train.labels))
    self.set_val(DataValues(x=mnist.test.images, y=mnist.test.labels))
    self.set_test(DataValues(x=mnist.test.images, y=mnist.test.labels))
