# -*- coding: utf-8 -*-
"""
@Time    : 4/30/19 10:46 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import tensorflow as tf
from python.cluster.data import DataSet
from python.cluster.weight import build_init_weight
from python.cluster.util import Parameter
import numpy as np

data_set = DataSet()
x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
    image = data_set.train_images[0].reshape(-1, 1)
    indices = np.concatenate((np.arange(image.shape[0]).reshape(-1, 1), np.zeros(image.shape[0]).reshape(-1, 1)),
                             axis=1)
    values = image.reshape(-1)
    dense_shape = np.array([image.shape[0], image.shape[1]])
    sp = tf.SparseTensorValue(indices=indices, values=values, dense_shape=dense_shape)
    print(sess.run(y, feed_dict={x: sp}))
