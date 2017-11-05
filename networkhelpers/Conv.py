import tensorflow as tf 

from tensorflow.python.ops import control_flow_ops

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
  def conv_layer(
    val, shape, is_training = tf.constant(False)
    ):
    """
    Creates a convolutional layer given the input and shape.
    """
    bias_param = shape[3]

    W = Conv2d.weight_variable(shape)
    b = Conv2d.bias_varible([bias_param])

    layer = Conv2d.conv2d(val, W) + b
    batch_norm = Conv2d.batch_norm(layer, bias_param, is_training)

    return tf.nn.elu(batch_norm)

  @staticmethod
  def full_layer(val, size):
    """
    Creates the full conv layer.
    """
    in_size = int(val.get_shape()[1])
    W = Conv2d.weight_variable([in_size, size])
    b = Conv2d.bias_varible([size])
    return tf.matmul(val, W) + b

  @staticmethod
  def batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0,
                                        dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0,
                                          dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out],
                            initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out],
                              initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2],
                                          name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))
    normed = tf.nn.batch_norm_with_global_normalization(x,
              mean, var, beta, gamma, 1e-3, True)
    return normed
