# -*- coding: utf-8 -*-
"""
@Time    : 4/25/19 3:17 PM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
"""gabor滤波处理后并保存特征向量"""
import pandas as pd
import numpy as np
from python.cluster.filter import layer1_filters
from python.cluster.util import DataSet
import tensorflow as tf
from tqdm import tqdm

BATCH_NUM = 10000


def gabor_process(dataset, type):
    data_len = dataset.test_data_len if type == dataset.TYPE_TEST else dataset.train_data_len
    outs = None
    with tf.Session() as sess:
        X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        # layer1 shape [-1,14,14,GABOR_KERNEL_SIZE_NUM*GABOR_KERNEL_DIRECTION_NUM]
        layer1_feature_map = []
        for flt in layer1_filters:
            convolution_map = tf.nn.conv2d(X, flt, strides=[1, 1, 1, 1], padding='SAME')
            max_pool_map = tf.nn.max_pool(convolution_map, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                                          padding='SAME')
            layer1_feature_map.append(max_pool_map)
        layer1_feature_map = tf.concat(layer1_feature_map, -1)
        layer_flatten = tf.layers.flatten(layer1_feature_map)
        for i in tqdm(range(int(data_len / BATCH_NUM))):
            images, labels = dataset.test_next_batch(
                BATCH_NUM) if type == dataset.TYPE_TEST else dataset.train_next_batch(BATCH_NUM)
            out = sess.run(layer_flatten, feed_dict={X: images})
            if outs is None:
                outs = out
            else:
                outs = np.concatenate([outs, out], axis=0)
    return outs


if __name__ == '__main__':
    dataset = DataSet()
    test_images = gabor_process(dataset, DataSet.TYPE_TEST)
    train_images = gabor_process(dataset, DataSet.TYPE_TRAIN)

    train_csv = pd.DataFrame(train_images)
    train_csv["target"] = dataset.train_labels
    train_csv.to_csv("../data/train_data_gabor.csv", index=False)

    test_csv = pd.DataFrame(test_images)
    test_csv["target"] = dataset.test_labels
    test_csv.to_csv("../data/test_data_gabor.csv", index=False)
