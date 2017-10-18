import sys

from collections import namedtuple

# representing the (train, label) pair of train, val and test data.
DataValues = namedtuple("DataValues", ["x", "y"])

class Dataset(object):
  """
  Dataset class
  """

  def __init__(self, batch_size):
    self.batch_size = batch_size
    self._train = None
    self._val = None
    self._test = None

  
  def next_batch(self, dataset):
    """
      A generator function that yields the batch based on the batch size specified.
    """
    # print(dataset.x.shape[0])
    # sys.exit(1)
    dataset_len = dataset.x.shape[0]
    for index in range(dataset_len // self.batch_size):
      start = index * self.batch_size 
      end = start + self.batch_size
      yield dataset.x[start: end], dataset.y[start: end]


  
  def set_train(self, train):
    """
    Sets the training set
    """
    self._train = train

  def get_train(self):
    """
    Returns the training data 
    """
    if self._train is None:
      raise NotImplementedError("Should Be Implemented in Subclasses")
    return self._train

  def set_val(self, val):
    """
    Sets the validation set property
    """
    self._val = val

  def get_val(self):
    """
    Returns the validaiton data of the object.
    """
    if self._val is None:
      raise NotImplementedError("Should be implemented in Subclasses")
    return self._val

  def set_test(self, test):
    """
    Set test set
    """
    self._test = test
  
  def get_test(self):
    """
    Returns the test data.
    """
    if self._test is None:
      raise NotImplementedError("Should be implemented in the Subclass")
    return self._test






