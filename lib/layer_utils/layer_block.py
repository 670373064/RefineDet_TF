import tensorflow as tf
import numpy as np


def normalize_layer(bottom, scalar, name, eps=1e-5):
    with tf.name_scope(name):
        channel_square = tf.pow(bottom, 2, name='input_square')
        channel_sum = tf.reduce_sum(channel_square, axis=[-1], keep_dims=True, name='channel_sum')
        channel_normalize = tf.sqrt(channel_sum, name='channel_sum_sqrt') + eps
        output1 = tf.divide(bottom, channel_normalize, name='channel_normalize')
        init_vec = np.ones(shape=[1, 1, 1, bottom.get_shape().as_list()[-1]], dtype=np.float32) * scalar
        scalar_weights = tf.get_variable('scalar_weight', shape=[1, 1, 1, bottom.get_shape().as_list()[-1]],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(init_vec))
        output = tf.multiply(output1, scalar_weights, name='scalar_output')
    return output, scalar_weights