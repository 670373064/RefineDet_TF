import tensorflow as tf
import tensorlayer as tl
import os, sys, time
import numpy as np
from layer_utils.layer_block import normalize_layer


class Vgg(object):
    def __init__(self, input_size, channel, layers, mean):
        self.input_size = input_size
        self.channel = channel
        self.layers = layers
        self.input_x = tf.placeholder(shape=[None, input_size, input_size, channel], dtype=tf.float32, name='input_img')
        self.mean = mean

    def vgg16(self):
        network = tl.layers.InputLayer(self.input_x, name='input_layer')
        with tf.name_scope('vgg16_preprocess') as scope:
            mean = tf.constant(self.mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            network.outputs = network.outputs - mean
        # conv1
        network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                                      name='pool1')
        # conv2  stride 2
        network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

        # conv3  stride 4
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

        # conv4  stride 8
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')

        # conv5  stride 16
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')
        return network

    def origin_fc_layers(self, network):
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
        network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
        network = tl.layers.DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
        return network

    def vgg16_fc_conv(self, network):
        # fc6
        network = tl.layers.AtrousConv2dLayer(network, n_filter=1024, filter_size=(3, 3), rate=3, padding='SAME',
                                              act=tf.nn.relu, name='fc6')
        # fc7
        network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(1, 1), padding='SAME',
                                   act=tf.nn.relu, name='fc7')
        return network

    def extra_layer(self, network):
        # conv6 stride 32
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1, 1), padding='SAME', strides=(1, 1),
                                   act=tf.nn.relu, name='conv6_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), padding='SAME', strides=(2, 2),
                                   act=tf.nn.relu, name='conv6_2')
        return network

    def refinement_layer(self, network):
        # refined layer
        arm_source_layers = ['conv4_3', 'conv5_3', 'fc7', 'conv6_2']
        normalizations = [10, 8, -1, -1]
        network, scalar_weights_43 = normalize_layer(network, 10, name='conv4_3_normalize_layer')
        network.all_params.append(scalar_weights_43)
        arm_source_layers.reverse()
        normalizations.reverse()
        for index, layer in enumerate(arm_source_layers):
            if normalizations[index] != -1:
                pass
        return network

    def vgg16_load_weights(self, sess, network, params_path):
        if not os.path.isfile(os.path.join(params_path, "vgg16_weights.npz")):
            raise ValueError('vgg weights file not found')
        npz = np.load(os.path.join(params_path, "vgg16_weights.npz"))
        params = []
        for val in sorted(npz.items()):
            if len(val[1].shape) == 2:
                break
            print("  Loading %s" % str(val[1].shape))
            params.append(val[1])
        tl.files.assign_params(sess, params, network)


if __name__ == '__main__':
    vgg16 = Vgg(224, 3, 16, [102.9801, 115.9465, 122.7717])
    network = vgg16.vgg16()
    network = vgg16.vgg16_fc_conv(network)
    network = vgg16.extra_layer(network)
    print(type(network.all_layers))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    vgg16.vgg16_load_weights(sess, network, '/home/aurora/backup')
    sess.run(tf.global_variables_initializer())
    with sess:
        network.print_params()
        network.print_layers()



