# -*- coding: utf-8 -*-
"""
@Time    : 4/30/19 11:31 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""

import numpy as np
from python.cluster.util import Parameter
import tensorflow as tf


def build_init_weight():
    rows = Parameter.middle_neuron_num
    cols = Parameter.input_neuron_num
    length = rows * cols

    # 挑选稀疏矩阵的下标
    index = np.random.choice(length, size=int(length * Parameter.input_weight_density), replace=False)
    # 转换为二维形式
    indices = np.zeros((index.shape[0], 2), dtype=np.uint32)
    indices[:, 0] = index / cols
    indices[:, 1] = index % cols

    weight_input2cluster = tf.SparseTensor(indices=indices, values=np.ones(len(index), dtype=np.float32),
                                           dense_shape=[rows, cols])
    return weight_input2cluster


if __name__ == '__main__':
    t = build_init_weight()
