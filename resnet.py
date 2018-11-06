# -- coding: UTF-8 --
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import tensorflow.contrib.slim as slim

from tensorflow.python.training import moving_averages


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, clips, labels, mode, use_nonlocal, gup_id=0):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          clips: Batches of clips . [batch_size, frames, crop_size, crop_size, channels]
          labels: Batches of labels . [batch_size, num_classes]
          mode: One of 'train' and 'test'
          use_nonlocal: One of 'use_nonlocal' and 'no_nonlocal'
        """
        self.hps = hps
        self._clips = clips
        self.labels = labels
        self.mode = mode
        self.use_nonlocal = use_nonlocal
        self.gpu_id = gup_id
        self._extra_train_ops = []

    # build graph and return nodes
    def build_graph(self):

        logits, predictions, cost = self._build_model()
        grads, bn_ops = self._get_grads()
        return logits, predictions, cost, bn_ops

    # build_model
    def _build_model(self):
        with tf.variable_scope('scale1'):
            x = self._clips
            x = self._conv3d('conv1', x, [5,7,7], 3, 64, self._stride_arr([1,2,2]))
            x = self._batch_norm('conv1_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
        print(x.shape)
        x = tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # configs
        activate_before_residual = [True, False, False, False]
        res_func = self._bottleneck_residual
        filters = [64, 256, 512, 1024, 2048]
        block_num = [3, 4, 6, 3]

        # res2
        with tf.variable_scope('scale2'):
            with tf.variable_scope('block1'):
                x = res_func(x, filters[0], filters[1],
                           self._stride_arr([1,1,1]),
                           activate_before_residual[0],
                           inflate=True)
            for i in six.moves.range(1, block_num[0]):
                with tf.variable_scope('block%d' % (i+1)):
                    x = res_func(x, filters[1], filters[1], self._stride_arr([1,1,1]), False, inflate=True)
        print(x.shape)
        x = tf.nn.max_pool3d(x, ksize=[1, 3, 1, 1, 1], strides=[1, 2, 1, 1, 1],
                               padding='SAME', name='pool2')
        # res3
        with tf.variable_scope('scale3'):
            with tf.variable_scope('block1'):
                x = res_func(x, filters[1], filters[2],
                           self._stride_arr([1,1,1]),
                           activate_before_residual[1],
                           inflate=True)
            for i in six.moves.range(1, block_num[1]):
                with tf.variable_scope('block%d' % (i+1)):
                    if i%2:
                        x = res_func(x, filters[2], filters[2], self._stride_arr([1,1,1]), False, inflate=False)
                        if self.use_nonlocal == 'use_nonlocal':
                            x = self._nonlocal(x, out_channels=512, name='NonLocalBlock')
                    else:
                        x = res_func(x, filters[2], filters[2], self._stride_arr([1,1,1]), False, inflate=True)
        print(x.shape)

        # res4
        with tf.variable_scope('scale4'):
            with tf.variable_scope('block1'):
                x = res_func(x, filters[2], filters[3],
                           self._stride_arr([1,1,1]),
                           activate_before_residual[2],
                           inflate=True)
            for i in six.moves.range(1, block_num[2]):
                with tf.variable_scope('block%d' % (i+1)):
                    if i%2:
                        x = res_func(x, filters[3], filters[3], self._stride_arr([1,1,1]), False, inflate=False)
                        if self.use_nonlocal == 'use_nonlocal':
                            x = self._nonlocal(x, out_channels=1024, name='NonLocalBlock')
                    else:
                        x = res_func(x, filters[3], filters[3], self._stride_arr([1,1,1]), False, inflate=True)
        print(x.shape)

        # res5
        with tf.variable_scope('scale5'):
            with tf.variable_scope('block1'):
                x = res_func(x, filters[3], filters[4],
                           self._stride_arr([1,1,1]),
                           activate_before_residual[3],
                           inflate=False)
            for i in six.moves.range(1, block_num[3]):
                with tf.variable_scope('block%d' % (i+1)):
                    if i%2:
                        x = res_func(x, filters[4], filters[4], self._stride_arr([1,1,1]), False, inflate=True)
                    else:
                        x = res_func(x, filters[4], filters[4], self._stride_arr([1,1,1]), False, inflate=False)
        print(x.shape)
        x = tf.nn.avg_pool3d(x, ksize=[1, 4, 7, 7, 1], strides=[1, 1, 1, 1, 1],
                               padding='VALID', name='pool5')
        print(x.shape)
        if self.mode == 'train':
            x = tf.nn.dropout(x, keep_prob=0.5)

        # glob_avg_pooling when test
        if self.mode == 'test':
            with tf.variable_scope('averagePool'):
                x = self._global_avg_pool(x)
                print(x.shape)

        # fc + Softmax
        with tf.variable_scope('fc'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        # costs
        with tf.variable_scope('costs'):
            # cross_entropy
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            # L2_loss
            self.cost += self._decay()
            tf.summary.scalar('loss_gpu%d'%self.gpu_id, self.cost)
        return logits, self.predictions, self.cost
    def _get_grads(self):
        # extra_train_ops for bn
        bn_ops = self._extra_train_ops
        return [], bn_ops

    def _stride_arr(self, stride):
        return [1, stride[0], stride[1], stride[2], 1]


    def _nonlocal(self, input_x, out_channels, name='NonLocalBlock'):
        batchsize, time, height, width, in_channels = input_x.get_shape().as_list()
        with tf.variable_scope('NonLocalBlock'):
            with tf.variable_scope('g'):
                g = self._conv3d('conv1', input_x, [1,1,1], out_channels, out_channels/2, [1, 1, 1, 1, 1])
            with tf.variable_scope('phi'):
                phi = self._conv3d('conv2', input_x, [1,1,1], out_channels, out_channels/2, [1, 1, 1, 1, 1])
            with tf.variable_scope('theta'):
                theta = self._conv3d('conv3', input_x, [1,1,1], out_channels, out_channels/2, [1, 1, 1, 1, 1])

            g_x = tf.reshape(g, [batchsize, time*height*width, out_channels / 2])
            theta_x = tf.reshape(theta, [batchsize, time*height*width, out_channels / 2])
            phi_x = tf.reshape(phi, [batchsize, time*height*width, out_channels / 2])
            phi_x = tf.transpose(phi_x, [0, 2, 1])

            f = tf.matmul(theta_x, phi_x)
            f_softmax = tf.nn.softmax(f, -1)
            y = tf.matmul(f_softmax, g_x)
            y = tf.reshape(y, [batchsize, time, height, width, out_channels / 2])

            with tf.variable_scope('w'):
                w_y = self._conv3d('conv4', y, [1,1,1], out_channels / 2, out_channels, [1, 1, 1, 1, 1])
                w_y = self._batch_norm('bn', w_y)
        z = input_x + w_y
        return z


    # bottleneck resnet block
    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False, inflate=False):
        orig_x = x
        # a
        with tf.variable_scope('a'):
            if inflate:
                x = self._conv3d('conv1', x, [3,1,1], in_filter, out_filter/4, stride)
            else:
                x = self._conv3d('conv1', x, [1,1,1], in_filter, out_filter/4, stride)
            x = self._batch_norm('bn1', x)
            x = self._relu(x, self.hps.relu_leakiness)

        # b
        with tf.variable_scope('b'):
            if in_filter != out_filter and out_filter != 256:
                x = self._conv3d('conv2', x, [1,3,3], out_filter/4, out_filter/4, [1, 1, 2, 2, 1])
            else:
                x = self._conv3d('conv2', x, [1,3,3], out_filter/4, out_filter/4, [1, 1, 1, 1, 1])
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)

        # c
        with tf.variable_scope('c'):
            x = self._conv3d('conv3', x, [1,1,1], out_filter/4, out_filter, [1, 1, 1, 1, 1])
            x = self._batch_norm('bn3', x)

        # when channels change, shortcut
        with tf.variable_scope('shortcut'):
            if in_filter != out_filter and out_filter != 256:
                orig_x = self._conv3d('project', orig_x, [1,1,1], in_filter, out_filter, [1, 1, 2, 2, 1])
                orig_x = self._batch_norm('bn4', orig_x)
            elif in_filter != out_filter:
                orig_x = self._conv3d('project', orig_x, [1,1,1], in_filter, out_filter, [1, 1, 1, 1, 1])
                orig_x = self._batch_norm('bn4', orig_x)
        x += orig_x
        x = self._relu(x, self.hps.relu_leakiness)

        tf.logging.info('image after unit %s', x.get_shape())
        return x


    # Batch Normalization
    def _batch_norm(self, name, x):

        params_shape = [x.get_shape()[-1]]
        # offset
        beta = tf.get_variable('beta',
                             params_shape,
                             tf.float32,
                             initializer=tf.constant_initializer(0.0, tf.float32))
        # scale
        gamma = tf.get_variable('gamma',
                              params_shape,
                              tf.float32,
                              initializer=tf.constant_initializer(1.0, tf.float32))

        if self.mode == 'train':
            mean, variance = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
            moving_mean = tf.get_variable('moving_mean',
                                          params_shape, tf.float32,
                                          initializer=tf.constant_initializer(0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance',
                                              params_shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)
            # moving_mean = moving_mean * decay + mean * (1 - decay)
            # moving_variance = moving_variance * decay + variance * (1 - decay)
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                            moving_mean, mean, 0.9))
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                            moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable('moving_mean',
                                   params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32),
                                   trainable=False)
            variance = tf.get_variable('moving_variance',
                                       params_shape, tf.float32,
                                       initializer=tf.constant_initializer(1.0, tf.float32),
                                       trainable=False)
            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)

        # BN：((x-mean)/var)*gamma+beta
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y


    # L2_loss
    def _decay(self):
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))
        tf.summary.scalar('l2_loss_gpu%d' % self.gpu_id, tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs)))
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    # 3D_conv
    def _conv3d(self, name, x, filter_size, in_filters, out_filters, strides):
        # filter: [filter_depth, filter_height, filter_width]
        # strides: [1, depth_stride, x_stride, y_stride, 1]
        n = filter_size[0]*filter_size[1]*filter_size[2]*out_filters
        kernel = tf.get_variable(
              'weights',
              [filter_size[0], filter_size[1], filter_size[2], in_filters, out_filters],
              tf.float32,
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv3d(x, kernel, strides, padding='SAME')

    # leaky ReLU
    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    # fc
    def _fully_connected(self, x, out_dim):
        # reshape
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            initializer=tf.variance_scaling_initializer(distribution="uniform"))

        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, w, b)
        print(x.shape)
        return x

    # _global_avg_pool
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 5
        return tf.reduce_mean(x, [1, 2, 3], keepdims=True)