import tensorflow as tf
from layer_utils.layer_block import normalize_layer
import numpy as np


def normalize_layer_test():
    arr = np.array([[[[2, 2],
                      [3, 3]],
                      [[4, 4],
                      [1, 1]]]])
    inputs = tf.constant(arr, dtype=tf.float32)
    scalar = 8
    name = 'test'
    output = normalize_layer(inputs, scalar, name)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output, scalar_weight = sess.run(output)
    print(output)
    print(scalar_weight)


if __name__ == '__main__':
    normalize_layer_test()

