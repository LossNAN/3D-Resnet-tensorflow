# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
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
import input_test_data
import math
import numpy as np
from i3d_nonlocal import InceptionI3d
from i3d_utils import *
from tensorflow.python import pywrap_tensorflow
from resnet import ResNet
from collections import namedtuple

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 4
flags.DEFINE_integer('batch_size', 6, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 256, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 400, 'The num of class')
flags.DEFINE_integer('block_num', 0, 'The num of nonlocal block')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

HParams = namedtuple('HParams',
					 ['batch_size', 'num_classes', 'use_bottleneck', 'weight_decay_rate', 'relu_leakiness'])
hps = HParams(FLAGS.batch_size, FLAGS.classics, True, FLAGS.weight_decay, 0)

def run_training():
	# Get the sets of images and labels for training, validation, and
	# Tell TensorFlow that the model will be built into the default Graph.
	pre_model_save_dir = './models/4GPU_sgd0block_i3d_400000_6_64_0.0001_decay'

	video_path_list = np.load('./data_list/data_list.npy')
	id_list = np.load('./data_list/id_list.npy')
	labels = np.load('./data_list/label_list.npy')
	
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		train_input_queue = tf.train.slice_input_producer([video_path_list, id_list], num_epochs=1, shuffle=False)
		video_path = train_input_queue[0]
		train_ids = train_input_queue[1]

		rgb_train_images, _, _ = tf.py_func(func=input_test_data.get_frames,
				   inp=[video_path[0],video_path[1],FLAGS.num_frame_per_clib,FLAGS.crop_size,False],
				   Tout=[tf.float32, tf.double, tf.int64],
				   )

		batch_videos, batch_ids = tf.train.batch([rgb_train_images, train_ids], batch_size=FLAGS.batch_size*gpu_num, capacity=200,
												num_threads=20, shapes=[(FLAGS.num_frame_per_clib,FLAGS.crop_size,FLAGS.crop_size,3), ()])

		norm_score = []
		with tf.variable_scope(tf.get_variable_scope()):
			for gpu_index in range(0, gpu_num):
				with tf.device('/gpu:%d' % gpu_index):
					with tf.name_scope('GPU_%d' % gpu_index) as scope:
						NET = ResNet(hps,
									 batch_videos[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :,:],
									 batch_ids[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
									 'test',
									 'no_nonlocal',
									 gup_id=gpu_index)
						logit, predictions, rgb_loss, ops = NET.build_graph()
						tf.get_variable_scope().reuse_variables()
					norm_score.append(predictions)
		norm_score = tf.concat(norm_score, 0)
		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

	ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print ("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		print ("load complete!")

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess, coord)
	predicts = []
	ids = []
	print("Start! test begin......queune and network init, this will take a few minutes, please wait..........")
	try:
		while not coord.should_stop():
			start_time = time.time()
			predict, video_id = sess.run([norm_score, batch_ids])
			predicts.extend(predict)
			ids.extend(video_id)
			duration = time.time() - start_time
			print('Test_step: %d/%d , time use: %.3f' % (video_id[-1], len(labels), duration))
	except tf.errors.OutOfRangeError: 
		print("Test done! kill all the threads....")
	finally:
		coord.request_stop()
		print('all threads are asked to stop!')
	coord.join(threads)
	np.save('./result/predicts.npy', predicts)
	np.save('./result/ids.npy', ids)
	np.save('./result/labels.npy', labels)
	topk(predicts, labels, ids)
	print('done!')


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()
