# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import sys
sys.path.append('../../')
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import math
import numpy as np
from i3d_utils import *
from tensorflow.python import pywrap_tensorflow
import Train as train_net
from resnet import ResNet
from collections import namedtuple

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 4
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 400000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 6, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('sample_rate', 8, 'Sample rate for clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 400, 'The num of class')
flags.DEFINE_integer('block_num', 0, 'The num of nonlocal block')
flags.DEFINE_bool('use_nonlocal', True, 'use or not nonlocal')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
FLAGS = flags.FLAGS

pre_model_save_dir = '../../checkpoints/resnet_pretrain/inflated_50'
#pre_model_save_dir = './models/4GPU_sgd0block_i3d_400000_6_64_0.0001_decay'
model_save_dir = './models/%dGPU_sgd%dblock_i3d_400000_%d_64_0.0001_decay'%(gpu_num, FLAGS.block_num, FLAGS.batch_size)

HParams = namedtuple('HParams',
                     ['batch_size', 'num_classes', 'use_bottleneck', 'weight_decay_rate', 'relu_leakiness'])
hps = HParams(FLAGS.batch_size, FLAGS.classics, True, FLAGS.weight_decay, 0)

def run_training():
    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    video_path_list = np.load('./data_list/train_data_list.npy')
    label_list = np.load('./data_list/train_label_list.npy')
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        train_input_queue = tf.train.slice_input_producer([video_path_list, label_list], shuffle=True)
        video_path = train_input_queue[0]
        train_label = train_input_queue[1]

        rgb_train_images, _, _ = tf.py_func(func=input_data.get_frames,
                   inp=[video_path,FLAGS.num_frame_per_clib,FLAGS.crop_size,FLAGS.sample_rate,False],
                   Tout=[tf.float32, tf.double, tf.int64],
                   )

        batch_videos, batch_labels = tf.train.batch([rgb_train_images, train_label], batch_size=FLAGS.batch_size*gpu_num, capacity=200,
                                                num_threads=20, shapes=[(FLAGS.num_frame_per_clib/FLAGS.sample_rate,FLAGS.crop_size,FLAGS.crop_size,3), ()])

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=150000, decay_rate=0.1,
                                                   staircase=True)
        opt_rgb = tf.train.MomentumOptimizer(learning_rate, 0.9)

        tower_grads = []
        logits = []
        loss = []
        bn_ops = []
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_index in range(0, gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('GPU_%d' % gpu_index):
                        NET = ResNet(hps,
                                     batch_videos[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :,:],
                                     batch_labels[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                                     'train',
                                     'no_nonlocal',
                                     gup_id=gpu_index)
                        logit, predictions, rgb_loss, ops = NET.build_graph()
                        bn_ops.extend(ops)
                        tf.get_variable_scope().reuse_variables()
                        rgb_grads = opt_rgb.compute_gradients(rgb_loss)
                    tower_grads.append(rgb_grads)
                    logits.append(logit)
                    loss.append(rgb_loss)
        logits = tf.concat(logits, 0)
        accuracy = tower_acc(logits, batch_labels)
        grads = average_gradients(tower_grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_rgb = opt_rgb.apply_gradients(grads, global_step=global_step)
            train_op = tf.group(apply_gradient_rgb, bn_ops)
            null_op = tf.no_op()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        i3d_map = {}
        for variable in tf.trainable_variables():
            if 'fc' not in variable.name and 'NonLocalBlock' not in variable.name:
            #if 'NonLocalBlock' not in variable.name:
                i3d_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=i3d_map, reshape=True)

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Create summary writter
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('error', 1.0-accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    train_writer = tf.summary.FileWriter('./visual_logs/%dGPU_sgd%dblock_train_i3d_400000_%d_64_0.0001_decay'%(gpu_num, FLAGS.block_num, FLAGS.batch_size), sess.graph)
    #test_writer = tf.summary.FileWriter('./visual_logs/%dGPU_sgd%dblock_test_i3d_400000_%d_64_0.0001_decay'%(gpu_num, FLAGS.batch_size, FLAGS.block_num), sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    
    for step in range(FLAGS.max_steps):
        start_time = time.time()
        sess.run(train_op)
        duration = time.time() - start_time
        print('Step %d: %.3f sec, end time : after %.3f days' % (step, duration, (FLAGS.max_steps-step)*duration/86400))
        
        if step % 10 == 0 or (step + 1) == FLAGS.max_steps:
            print('Training Data Eval:')
            summary, acc, loss_rgb = sess.run([merged, accuracy, loss])
            print("accuracy: " + "{:.5f}".format(acc))
            print("rgb_loss: " + "{:.5f}".format(np.mean(loss_rgb)))
            train_writer.add_summary(summary, step)
            
        if (step+1) % 20000 == 0 or (step + 1) == 400000:
            saver.save(sess, os.path.join(model_save_dir, 'nonlocal_kinetics_model'), global_step=step)

    coord.request_stop()
    coord.join(threads)
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()