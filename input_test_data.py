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

"""Functions for load test data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


def get_data(filename, num_frames_per_clip=64, s_index=0):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames)==0:
            print('DATA_ERRO: %s'%filename)
            return [], s_index, True
        if (len(filenames)-s_index) <= num_frames_per_clip:
            filenames = sorted(filenames)
            if len(filenames) < num_frames_per_clip:
                for i in range(num_frames_per_clip):
                    if i >= len(filenames):
                        i = len(filenames)-1
                    image_name = str(filename) + '/' + str(filenames[i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            else:
                for i in range(num_frames_per_clip):
                    image_name = str(filename) + '/' + str(filenames[len(filenames)-num_frames_per_clip+i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            return ret_arr, s_index, True
    filenames = sorted(filenames)
    for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, s_index, False


def get_frames_data(filename, num_frames_per_clip, s_index, add_flow):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    filename_i = os.path.join(filename, 'i')
    rgb_ret_arr, _ , is_end= get_data(filename, num_frames_per_clip, s_index)
    if not add_flow:
        return rgb_ret_arr, [], s_index, is_end
    filename_x = os.path.join(filename, 'x')
    flow_x, _, _ = get_data(filename_x, num_frames_per_clip, s_index)
    flow_x = np.expand_dims(flow_x, axis=-1)
    filename_y = os.path.join(filename, 'y')
    flow_y, _, _ = get_data(filename_y, num_frames_per_clip, s_index)
    flow_y = np.expand_dims(flow_y, axis=-1)
    flow_ret_arr = np.concatenate((flow_x, flow_y), axis=-1)
    return rgb_ret_arr, flow_ret_arr, s_index, is_end


def get_frames(filename, s_index, num_frames_per_clip, crop_size, add_flow):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    rgb_ret_arr, _s_index, _ = get_data(filename, num_frames_per_clip, int(s_index))
    rgb_ret_arr = data_process(rgb_ret_arr, crop_size)
    if not add_flow:
        return rgb_ret_arr, [], _s_index


def data_process(tmp_data, crop_size):
    img_datas = []
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)
    return img_datas


def read(filename, batch_size, s_index, num_frames_per_clip=64, crop_size=224, add_flow=False):
    rgb_data = []
    flow_data = []
    label = []
    line = filename.strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    print("Loading a video clip from {}...strat_frame = {}".format(dirname, s_index))
    tmp_rgb_data, tmp_flow_data, _, is_end = get_frames_data(dirname, num_frames_per_clip, s_index, add_flow)
    if len(tmp_rgb_data) != 0:
        rgb_img_datas = data_process(tmp_rgb_data, crop_size)
        if add_flow:
            flow_img_datas = data_process(tmp_flow_data, crop_size)
            flow_data.append(flow_img_datas)
        rgb_data.append(rgb_img_datas)
        label.append(int(tmp_label))
    else:
        return [], [], [], s_index + num_frames_per_clip, is_end


    np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
    np_arr_flow_data = np.array(flow_data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size), s_index+num_frames_per_clip, is_end


def read_clip_and_label(filenames, batch_size, s_index, step, num_frames_per_clip=64, crop_size=224, add_flow=False):
    s_index = s_index
    video_id = step
    rgb_data = []
    flow_data = []
    labels = []
    ids = []
    while True:
        if len(rgb_data) >= batch_size or video_id == len(filenames):
            break
        filename = filenames[video_id]
        data, flow, label, s_index, is_end = read(filename, 1, s_index, num_frames_per_clip=num_frames_per_clip, crop_size=crop_size, add_flow=add_flow)
        if len(data) != 0:
            id = video_id
            rgb_data.append(data)
            flow_data.append(flow)
            ids.append(id)
            if is_end:
                labels.extend(label)
        if is_end:
            video_id += 1
            s_index = 0
    # pad (duplicate) data/label if less than batch_size
    valid_len = len(rgb_data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            rgb_data.append(data)
            flow_data.append(flow)
            ids.append(int(id))
    np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
    np_arr_flow_data = np.array(flow_data).astype(np.float32)
    np_arr_label = np.array(labels).astype(np.int64)
    np_arr_ids = np.array(ids).astype(np.int64)

    return np_arr_rgb_data.reshape(batch_size, num_frames_per_clip, crop_size, crop_size, 3),\
           np_arr_flow_data, np_arr_label, \
           np_arr_ids.reshape(batch_size), s_index, video_id
