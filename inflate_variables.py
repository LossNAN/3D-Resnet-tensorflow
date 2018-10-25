import os
import tensorflow as tf
import re
import numpy as np
from tensorflow.python import pywrap_tensorflow
model_dir = './checkpoints/resnet_pretrain/50'
inflated_model_dir = './checkpoints/resnet_pretrain/inflated_50'
checkpoint_path = os.path.join(model_dir, "ResNet-L50.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
for name in sorted(var_to_shape_map):
    if 'weights' in name:
        print(name)
with tf.Session() as sess:
    for name in sorted(var_to_shape_map):
        variable = reader.get_tensor(name)
        new_variable = []
        if 'weights' in name:
            if 'scale1' in name:
                for i in range(5):
                    new_variable.append(variable/5)
                new_variable = np.array(new_variable)
                tf.Variable(initial_value=new_variable, name=name)
                print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
            elif 'scale2' in name:
                if 'a/' in name:
                    for i in range(3):
                        new_variable.append(variable/3)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
                else:
                    for i in range(1):
                        new_variable.append(variable/1)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
            elif 'scale3' in name:
                if 'a/' in name:
                    if int(name.split('/')[1][-1])%2:
                        inflate_num = 3
                    else:
                        inflate_num = 1
                    for i in range(inflate_num):
                        new_variable.append(variable/inflate_num)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
                else:
                    for i in range(1):
                        new_variable.append(variable/1)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
            elif 'scale4' in name:
                if 'a/' in name:
                    if int(name.split('/')[1][-1])%2:
                        inflate_num = 3
                    else:
                        inflate_num = 1
                    for i in range(inflate_num):
                        new_variable.append(variable/inflate_num)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
                else:
                    for i in range(1):
                        new_variable.append(variable/1)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
            elif 'scale5' in name:
                if 'a/' in name:
                    if int(name.split('/')[1][-1])%2:
                        inflate_num = 1
                    else:
                        inflate_num = 3
                    for i in range(inflate_num):
                        new_variable.append(variable/inflate_num)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
                else:
                    for i in range(1):
                        new_variable.append(variable/1)
                    new_variable = np.array(new_variable)
                    tf.Variable(initial_value=new_variable, name=name)
                    print('inflated %s, shape: %s to shape %s'%(name, variable.shape, new_variable.shape))
        else:
            tf.Variable(initial_value=variable, name=name)
            print('keep_variable: %s, shape: %s' % (name, variable.shape))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(inflated_model_dir, 'inflated_model'))