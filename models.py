import tensorflow as tf
from ops import conv2d, linear, batchnorm, layernorm, upsample2x, downsample2x


class Generator():

    def __init__(self, k_size=3, smallest_unit_n=64):
        self.k_size = k_size
        self.smallest_unit_n = smallest_unit_n

    def _residual_block(self, x, n_out, is_train, name='residual'):
        with tf.variable_scope(name):
            with tf.variable_scope('shortcut'):
                x1 = upsample2x(x)
                x1 = conv2d(x1, n_out, self.k_size, 1, 'SAME')
            with tf.variable_scope('normal'):
                x2 = batchnorm(x, is_train, name='batchnorm_0')
                x2 = tf.nn.relu(x2)
                x2 = upsample2x(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_0')
                x2 = batchnorm(x2, is_train, name='batchnorm_1')
                x2 = tf.nn.relu(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_1')
            return x1 + x2

    def __call__(self, x, is_train=True, is_reuse=False):
        with tf.variable_scope('generator') as scope:
            if is_reuse:
                scope.reuse_variables()

            with tf.variable_scope('first'):
                x = linear(x, 4 * 4 * 8 * self.smallest_unit_n)
            x = tf.reshape(x, [-1, 4, 4, 8 * self.smallest_unit_n])

            for i, times in enumerate([8, 4, 2, 1]):
                x = self._residual_block(x, times * self.smallest_unit_n, is_train, 'residual_{}'.format(i))

            with tf.variable_scope('last'):
                x = batchnorm(x, is_train)
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
            with tf.variable_scope('shortcut'):
                x1 = downsample2x(x)
                x1 = conv2d(x1, n_out, self.k_size, 1, 'SAME')
            with tf.variable_scope('normal'):
                x2 = layernorm(x, name='layernorm_0')
                x2 = tf.nn.relu(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_0')
                x2 = layernorm(x2, name='layernorm_1')
                x2 = tf.nn.relu(x2)
                x2 = downsample2x(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_1')
            return x1 + x2

    def __call__(self, x, is_train=True, is_reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if is_reuse:
                scope.reuse_variables()

            with tf.variable_scope('first'):
                x = conv2d(x, self.smallest_unit_n, self.k_size, 1, 'SAME')

            for i, times in enumerate([2, 4, 8, 8]):
                x = self._residual_block(x, times * self.smallest_unit_n, 'residual_{}'.format(i))

            x = tf.reshape(x, [-1, 4 * 4 * 8 * self.smallest_unit_n])

            with tf.variable_scope('last'):
                x = linear(x, 1)

            return x
