# -*- coding: utf-8 -*-
"""
@Time    : 4/25/19 3:17 PM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

LAYER_ONE_KERNEL_SIZE_NUM = 8
LAYER_TWO_KERNEL_SIZE_NUM = 4
KERNEL_DIRECTION_NUM = 8

train_data = pd.read_csv("../data/train_data.csv", dtype=np.float32)
test_data = pd.read_csv("../data/test_data.csv", dtype=np.float32)
train_labels = train_data["target"]
train_images = train_data.drop("target", axis=1).values.reshape(-1, 28, 28, 1) / 255
test_labels = test_data["target"]
test_images = test_data.drop("target", axis=1).values.reshape(-1, 28, 28, 1) / 255


def build_filters(kernel_size, sigma, lambd, theta, same=1):
    filters = []
    for i in range(len(kernel_size)):
        for j in range(len(theta)):
            kernel = cv2.getGaborKernel((kernel_size[i], kernel_size[i]), sigma[i], theta[j], lambd[i], 0.5, 0,
                                        ktype=cv2.CV_32F)

            filter_3_temp = tf.expand_dims(kernel, -1)
            filter_3 = filter_3_temp
            for k in range(same - 1):
                filter_3 = tf.concat([filter_3, filter_3_temp], -1)
            filter_4 = tf.expand_dims(filter_3, -1)
            filters.append(filter_4)
    return filters


layer1_filters = build_filters(kernel_size=[3, 5, 7, 9, 11, 13, 15, 17],
                               sigma=np.arange(2, LAYER_ONE_KERNEL_SIZE_NUM + 2),
                               lambd=np.arange(2, LAYER_ONE_KERNEL_SIZE_NUM + 2),
                               theta=np.arange(0, np.pi, np.pi / KERNEL_DIRECTION_NUM))
layer2_filters = build_filters(kernel_size=[3, 5, 7, 9],
                               sigma=np.arange(2, LAYER_TWO_KERNEL_SIZE_NUM + 2),
                               lambd=np.arange(2, LAYER_TWO_KERNEL_SIZE_NUM + 2),
                               theta=np.arange(0, np.pi, np.pi / KERNEL_DIRECTION_NUM),
                               same=LAYER_ONE_KERNEL_SIZE_NUM * KERNEL_DIRECTION_NUM)
with tf.Session() as sess:
    # layer1 shape [-1,14,14,GABOR_KERNEL_SIZE_NUM*GABOR_KERNEL_DIRECTION_NUM]
    layer1_feature_map = []
    for flt in tqdm(layer1_filters):
        convolution_map = tf.nn.conv2d(test_images, flt, strides=[1, 1, 1, 1], padding='SAME')
        max_pool_map = tf.nn.max_pool(convolution_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer1_feature_map.append(max_pool_map)
    layer1_feature_map = tf.concat(layer1_feature_map, -1)
    #
    layer2_feature_map = []
    for flt in tqdm(layer2_filters):
        convolution_map = tf.nn.conv2d(layer1_feature_map, flt, strides=[1, 1, 1, 1], padding='SAME')
        max_pool_map = tf.nn.max_pool(convolution_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer2_feature_map.append(max_pool_map)
    layer2_feature_map = tf.concat(layer2_feature_map, -1)
    layer_flatten = tf.layers.flatten(layer2_feature_map)
    print(layer_flatten.eval())
