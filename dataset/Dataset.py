from collections import namedtuple

class Dataset(object):

  def __init__(self, batch_size):
    self.batch_size = batch_size
    self._train = None
    self._val = None
    self._test = None

  
  def next_batch(self):
    """
      A generator function that yields the batch based on the batch size specified.
    """
    raise NotImplementedError(
      "Should implement this in a subclass")
  
  @property
  def train(self):
    """
    Returns the training data 
    """
    if self._train == None:
      raise NotImplementedError("Should Be Implemented in Subclasses")
    return self._train

  @property
  def val(self):
    """
    Returns the validaiton data of the object.
    """
    if self._val == None:
      raise NotImplementedError("Should be implemented in Subclasses")
    return self._val

  def test(self):
    """
    Returns the test data.
    """
    if self._test == None:
      raise NotImplementedError("Should be implemented in the Subclass")
    return self._test






DataValues = namedtuple("DataArgs", ["train", "labels"])
