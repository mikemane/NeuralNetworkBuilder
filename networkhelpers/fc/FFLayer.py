import tensorflow as tf


class FFLayer(object):
  # Feed forward layer helper.

  @staticmethod
  def layer_batch_norm(x, n_out, is_training):
      beta_init = tf.constant_initializer(value=0.0,
                                          dtype=tf.float32)
      gamma_init = tf.constant_initializer(value=1.0,
                                            dtype=tf.float32)
      beta = tf.get_variable("beta", [n_out],
                              initializer=beta_init)
      gamma = tf.get_variable("gamma", [n_out],
                                initializer=gamma_init)
      batch_mean, batch_var = tf.nn.moments(x, [0],
                                            name='moments')
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op = ema.apply([batch_mean, batch_var])
      ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
      def mean_var_with_update():
          with tf.control_dependencies([ema_apply_op]):
              return tf.identity(batch_mean, tf.identity(batch_var)
      mean, var = control_flow_ops.cond(is_training,
          mean_var_with_update,
          lambda: (ema_mean, ema_var))
      x_r = tf.reshape(x, [-1, 1, 1, n_out])
      normed = tf.nn.batch_norm_with_global_normalization(x_r,
                mean, var, beta, gamma, 1e-3, True)
      return tf.reshape(normed, [-1, n_out])