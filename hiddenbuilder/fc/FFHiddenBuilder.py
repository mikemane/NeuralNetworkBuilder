import tensorflow as tf

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from hiddenbuilder.HiddenBuilder import HiddenBuilder


class FFHiddenBuilder(HiddenBuilder):
    """
    Fully connected hidden builder.
    """

    def build(self):
      hidden_sizes = self.hidden_sizes.weights
      counter = 1
      he_init = tf.contrib.layers.variance_scaling_initializer()
      layer_counter = "hidden{}"

      norm_params = {
        "is_training": self.is_training,
        "decay": 0.99,
        "updates_collections": None
        }

    # Testing on the training set, batch normalisation just made the training value worse.

      with tf.name_scope("ff_hidden"):
        with tf.contrib.framework.arg_scope(
                [fully_connected],
                weights_initializer=he_init,
                # normalizer_fn=batch_norm,
                normalizer_params=norm_params,
                weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)
                ):

          hidden = fully_connected(
              self.inputs,
              hidden_sizes[0],
              scope=layer_counter.format(counter)
          )
          for hidden_size in hidden_sizes[1:]:
              counter += 1
              scope = layer_counter.format(counter)
              hidden = fully_connected(hidden, hidden_size, scope=scope)
        
          logits = fully_connected(
              hidden,
              self.target_dim,
              activation_fn=None,
              scope="outputs"
          )
          return logits
