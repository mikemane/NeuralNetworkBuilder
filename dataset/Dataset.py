
class Dataset(object):

  def __init__(self, batch_size):
    self.batch_size = batch_size

  
  def next_batch(self):
    """
      A generator function that yields the batch based on the batch size specified.
    """
    raise NotImplementedError(
      "Should implement this in a subclass")
