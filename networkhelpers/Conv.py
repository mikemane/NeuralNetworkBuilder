import tensorflow as tf 


class Conv2d(object):
  """
  Class for generating CNN objects.
  """

  @staticmethod
  def weight_variable(shape):
    """
    Creating weight variables.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  @staticmethod
  def bias_varible(shape):
    """
    Creates the bias variable.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  @staticmethod
  def conv2d(x, W, strides = 1):
    """
    Creates the convolutional Neural Network.
    """
    return tf.nn.conv2d(x, W, [1, strides, strides, 1], padding="SAME")

  def max_pool_2x2(x):
    """
    Performs the max pool operation.
    """
    return tf.nn.max_pool(
      x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")

  @staticmethod
  def conv_layer(val, shape):
    """
    Creates a convolutional layer given the input and shape.
    """
    W = Conv2d.weight_variable(shape)
    b = Conv2d.bias_varible([shape[3]])
    return tf.nn.relu(Conv2d.conv2d(val, W) + b)

  @staticmethod
  def full_layer(val, size):
    """
    Creates the full conv layer.
    """
    in_size = int(val.get_shape()[1])
    W = Conv2d.weight_variable([in_size, size])
    b = Conv2d.bias_varible([size])
    return tf.matmul(val, W) + b