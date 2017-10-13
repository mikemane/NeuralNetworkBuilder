import tensorflow as tf

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from hiddenstrategy.HiddenStrategy import HiddenStrategy


class FeedForwardStrategy(HiddenStrategy):

    def build(self):
      hidden_sizes = self.network.config.hidden_sizes.weights
      counter = 1
      he_init = tf.contrib.layers.variance_scaling_initializer()
      layer_counter = "hidden{}"

      norm_params = {
        "is_training": self.network.is_training,
        "decay": 0.99,
        "updates_collections": None
        }

      with tf.name_scope("ff_hidden"):
        with tf.contrib.framework.arg_scope(
                [fully_connected],
                weights_initializer=he_init,
                # normalizer_fn=batch_norm,
                normalizer_params=norm_params,
                weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)
                ):

          hidden = fully_connected(
              self.network.inputs,
              hidden_sizes[0],
              scope=layer_counter.format(counter)
          )
          for hidden_size in hidden_sizes[1:]:
              counter += 1
              scope = layer_counter.format(counter)
              hidden = fully_connected(hidden, hidden_size, scope=scope)
        
          t_shape = self.network.config.target
          t_shape  = t_shape.cls if t_shape.cls != None else t_shape.shape[1]

          logits = fully_connected(
              hidden,
              t_shape,
              activation_fn=None,
              scope="outputs"
          )
          return logits
