import tensorflow as tf
from ops import conv2d, linear, batch_norm, layer_norm, avgpool2d


class Generator():

    def __init__(self, k_size=3, smallest_unit_n=64):
        self.k_size = k_size
        self.smallest_unit_n = smallest_unit_n

    def _residual_block(self, x, n_out, is_train, name='residual'):
        with tf.variable_scope(name):
            x1_shape = x.get_shape().as_list()
            x1 = tf.image.resize_bilinear(x, (x1_shape[1] * 2, x1_shape[2] * 2))
            x1 = conv2d(x1, n_out, self.k_size, 1, 'SAME')
            x2 = batch_norm(x, is_train)
            x2 = tf.nn.relu(x2)
            x2_shape = x2.get_shape().as_list()
            x2 = tf.image.resize_bilinear(x2, (x2_shape[1] * 2, x2_shape[2] * 2))
            x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME')
            x2 = batch_norm(x2, is_train)
            x2 = tf.nn.relu(x2)
            x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME')
            return x1 + x2

    def __call__(self, x, is_train=True):
        with tf.variable_scope('first'):
            x = linear(x, 4 * 4 * 8 * self.smallest_unit_n)
        x = tf.reshape(x, [-1, 4, 4, 8 * self.smallest_unit_n])

        for i, times in enumerate([8, 4, 2, 1]):
            x = self._residual_block(x, times * self.smallest_unit_n, is_train, 'residual_{}'.format(i))

        with tf.variable_scope('last'):
            x = batch_norm(x)
            x = tf.nn.relu(x)
            x = conv2d(x, 3, self.k_size, 1, 'SAME')

        x = tf.tanh(x)

        return x


class Discriminator():

    def __init__(self, k_size=3, smallest_unit_n=64):
        self.k_size = k_size
        self.smallest_unit_n = smallest_unit_n

    def _residual_block(self, x, n_out, name='residual'):
        with tf.variable_scope(name):
            x1 = avgpool2d(x, self.k_size, 2, 'SAME')
            x1 = conv2d(x1, n_out, self.k_size, 1, 'SAME')
            x2 = layer_norm(x)
            x2 = tf.nn.relu(x2)
            x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME')
            x2 = layer_norm(x2)
            x2 = tf.nn.relu(x2)
            x2 = avgpool2d(x2, self.k_size, 2, 'SAME')
            x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME')
            return x1 + x2

    def __call__(self, x, is_train=True):
        with tf.variable_scope('first'):
            x = conv2d(x, self.smallest_unit_n, self.k_size, 1, 'SAME')

        for i, times in enumerate([2, 4, 8, 8]):
            x = self._residual_block(x, times * self.smallest_unit_n, 'residual_{}'.format(i))

        x = tf.reshape(x, [-1, 4 * 4 * 8 * self.smallest_unit_n])

        with tf.variable_scope('last'):
            x = linear(x, 1)

        return x
