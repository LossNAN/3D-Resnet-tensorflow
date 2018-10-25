# -- coding: UTF-8 --
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode, use_nonlocal, gup_id=0):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images 图片. [batch_size, image_size, image_size, 3]
      labels: Batches of labels 类别标签. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode
    self.use_nonlocal = use_nonlocal
    self.gpu_id = gup_id


    self._extra_train_ops = []

  # 构建模型图
  def build_graph(self):

    logits, predictions, cost = self._build_model()
    grads, bn_ops = self._get_grads()
    return logits, predictions, cost, bn_ops


  # 构建模型
  def _build_model(self):
    with tf.variable_scope('scale1'):
      x = self._images
      """第一层卷积（3,3x3/1,16）"""
      x = self._conv3d('conv1', x, [5,7,7], 3, 64, self._stride_arr([1,2,2]))
      x = self._batch_norm('conv1_bn', x)
      print(x)
      x = self._relu(x, self.hps.relu_leakiness)
    print(x.shape)
    x = tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # 激活前置
    activate_before_residual = [True, False, False, False]
    if self.hps.use_bottleneck:
      # bottleneck残差单元模块
      res_func = self._bottleneck_residual
      nonlocal = self._nonlocal
      # 通道数量
      filters = [64, 256, 512, 1024, 2048]
      # Block 数量
      block_num = [3, 4, 6, 3]
    else:
      # 标准残差单元模块
      res_func = self._residual
      nonlocal = self._nonlocal
      # 通道数量
      filters = [64, 256, 512, 1024, 2048]
      # Block 数量
      block_num = [3, 4, 6, 3]

    # 第一组
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
    # 第二组
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
                  x = nonlocal(x, out_channels=512, name='NonLocalBlock')
            else:
              x = res_func(x, filters[2], filters[2], self._stride_arr([1,1,1]), False, inflate=True)
    print(x.shape)
    # 第三组
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
                  x = nonlocal(x, out_channels=1024, name='NonLocalBlock')
            else:
              x = res_func(x, filters[3], filters[3], self._stride_arr([1,1,1]), False, inflate=True)
    print(x.shape)

    # 第四组
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

    # 全局池化层
    with tf.variable_scope('averagePool'):
      x = self._global_avg_pool(x)
      print(x.shape)

    # 全连接层 + Softmax
    with tf.variable_scope('fc'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    # 构建损失函数
    with tf.variable_scope('costs'):
      # 交叉熵
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      # 加和
      self.cost = tf.reduce_mean(xent, name='xent')
      # L2正则，权重衰减
      self.cost += self._decay()
      # 添加cost总结，用于Tensorborad显示
      tf.summary.scalar('loss_gpu%d'%self.gpu_id, self.cost)
    return logits, self.predictions, self.cost
  # 构建训练操作
  def _get_grads(self):

    # 计算训练参数的梯度
    #trainable_variables = tf.trainable_variables()
    #grads = tf.gradients(self.cost, trainable_variables)

    bn_ops = self._extra_train_ops
    return [], bn_ops

  # 把步长值转换成tf.nn.conv2d需要的步长数组
  def _stride_arr(self, stride):    
    return [1, stride[0], stride[1], stride[2], 1]

  # 残差单元模块
  def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    # 第1子层
    with tf.variable_scope('sub1'):
      # 3x3卷积，使用输入步长，通道数(in_filter -> out_filter)
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 3x3卷积，步长为1，通道数不变(out_filter)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
      if in_filter != out_filter:
        # 均值池化，无补零
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        # 通道补零(第4维前后对称补零)
        orig_x = tf.pad(orig_x, 
                        [[0, 0], 
                         [0, 0], 
                         [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]
                        ])
      # 合并残差
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

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

          f_softmax = tf.nn.softmax(f, 0)
          y = tf.matmul(f_softmax, g_x)
          y = tf.reshape(y, [batchsize, time, height, width, out_channels / 2])

          with tf.variable_scope('w'):
              w_y = self._conv3d('conv4', y, [1,1,1], out_channels / 2, out_channels, [1, 1, 1, 1, 1])
              w_y = self._batch_norm('bn', w_y)
      z = input_x + w_y
      return z


  # bottleneck残差单元模块
  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False, inflate=False):
    orig_x = x
    # 第1子层
    with tf.variable_scope('a'):
      # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
      if inflate:
        x = self._conv3d('conv1', x, [3,1,1], in_filter, out_filter/4, stride)
      else:
        x = self._conv3d('conv1', x, [1,1,1], in_filter, out_filter/4, stride)
      x = self._batch_norm('bn1', x)
      x = self._relu(x, self.hps.relu_leakiness)

    # 第2子层
    with tf.variable_scope('b'):
      # 3x3卷积，步长为1，通道数不变(out_filter/4)
      if in_filter != out_filter and out_filter != 256:
        x = self._conv3d('conv2', x, [1,3,3], out_filter/4, out_filter/4, [1, 1, 2, 2, 1])
      else:
        x = self._conv3d('conv2', x, [1,3,3], out_filter/4, out_filter/4, [1, 1, 1, 1, 1])
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)

    # 第3子层
    with tf.variable_scope('c'):
      # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
      x = self._conv3d('conv3', x, [1,1,1], out_filter/4, out_filter, [1, 1, 1, 1, 1])
      # BN和ReLU激活
      x = self._batch_norm('bn3', x)
    # 合并残差层
    with tf.variable_scope('shortcut'):
      # 当通道数有变化时
      if in_filter != out_filter and out_filter != 256:
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
        orig_x = self._conv3d('project', orig_x, [1,1,1], in_filter, out_filter, [1, 1, 2, 2, 1])
        orig_x = self._batch_norm('bn4', orig_x)
      elif in_filter != out_filter:
        orig_x = self._conv3d('project', orig_x, [1,1,1], in_filter, out_filter, [1, 1, 1, 1, 1])
        orig_x = self._batch_norm('bn4', orig_x)
      # 合并残差
      x += orig_x
      x = self._relu(x, self.hps.relu_leakiness)

    tf.logging.info('image after unit %s', x.get_shape())
    return x


  # Batch Normalization批归一化
  # ((x-mean)/var)*gamma+beta
  def _batch_norm(self, name, x):
    #with tf.variable_scope(name):
      #       # 输入通道维数
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
        # 为每个通道计算均值、标准差
        mean, variance = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
        # 新建或建立测试阶段使用的batch均值、标准差
        moving_mean = tf.get_variable('moving_mean', 
                                      params_shape, tf.float32,
                                      initializer=tf.constant_initializer(0.0, tf.float32),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', 
                                          params_shape, tf.float32,
                                          initializer=tf.constant_initializer(1.0, tf.float32),
                                          trainable=False)
        # 添加batch均值和标准差的更新操作(滑动平均)
        # moving_mean = moving_mean * decay + mean * (1 - decay)
        # moving_variance = moving_variance * decay + variance * (1 - decay)
        self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                        moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                        moving_variance, variance, 0.9))
      else:
        # 获取训练中积累的batch均值、标准差
        mean = tf.get_variable('moving_mean', 
                               params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32),
                               trainable=False)
        variance = tf.get_variable('moving_variance', 
                                   params_shape, tf.float32,
                                   initializer=tf.constant_initializer(1.0, tf.float32),
                                   trainable=False)
        # 添加到直方图总结
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)

      # BN层：((x-mean)/var)*gamma+beta
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y


  # 权重衰减，L2正则loss
  def _decay(self):
    costs = []
    # 遍历所有可训练变量
    for var in tf.trainable_variables():
      #只计算标有“DW”的变量
      if var.op.name.find(r'weights') > 0:
        costs.append(tf.nn.l2_loss(var))
    tf.summary.scalar('l2_loss_gpu%d' % self.gpu_id, tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs)))
    # 加和，并乘以衰减因子
    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  # 2D卷积
  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      # 获取或新建卷积核，正态随机初始化
      kernel = tf.get_variable(
              'DW',
              [filter_size, filter_size, in_filters, out_filters],
              tf.float32, 
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      # 计算卷积
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  # 3D卷积
  def _conv3d(self, name, x, filter_size, in_filters, out_filters, strides):
    # filter: [filter_depth, filter_height, filter_width]
    # strides: [1, depth_stride, x_stride, y_stride, 1]
      n = filter_size[0]*filter_size[1]*filter_size[2]*out_filters
      # 获取或新建卷积核，正态随机初始化
      kernel = tf.get_variable(
              'weights',
              [filter_size[0], filter_size[1], filter_size[2], in_filters, out_filters],
              tf.float32,
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      # 计算卷积
      return tf.nn.conv3d(x, kernel, strides, padding='SAME')

  # leaky ReLU激活函数，泄漏参数leakiness为0就是标准ReLU
  def _relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
  
  # 全连接层，网络最后一层
  def _fully_connected(self, x, out_dim):
    # 输入转换成2D tensor，尺寸为[N,-1]
    x = tf.reshape(x, [self.hps.batch_size, -1])
    # 参数w，平均随机初始化，[-sqrt(3/dim), sqrt(3/dim)]*factor
    w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        initializer=tf.variance_scaling_initializer(distribution="uniform"))
    # 参数b，0值初始化
    b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    # 计算x*w+b
    return tf.nn.xw_plus_b(x, w, b)

  # 全局均值池化
  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 5
    # 在第2&3维度上计算均值，尺寸由WxH收缩为1x1
    x = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
    if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.5)
    else:
        x = tf.nn.dropout(x, keep_prob=1.0)
    return tf.reduce_mean(x, [1, 2, 3])