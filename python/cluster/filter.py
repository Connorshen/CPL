# -*- coding: utf-8 -*-
"""
@Time    : 4/26/19 9:50 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import cv2
import tensorflow as tf
import numpy as np

LAYER_ONE_KERNEL_SIZE_NUM = 4
LAYER_TWO_KERNEL_SIZE_NUM = 4
KERNEL_DIRECTION_NUM = 8


def build_filters(kernel_size, sigma, lambd, theta, channel=1):
    filters = []
    for i in range(len(kernel_size)):
        for j in range(len(theta)):
            kernel = cv2.getGaborKernel((kernel_size[i], kernel_size[i]), sigma[i], theta[j], lambd[i], 0.5, 0,
                                        ktype=cv2.CV_32F)

            filter_3_temp = tf.expand_dims(kernel, -1)
            filter_3 = filter_3_temp
            for k in range(channel - 1):
                filter_3 = tf.concat([filter_3, filter_3_temp], -1)
            filter_4 = tf.expand_dims(filter_3, -1)
            filters.append(filter_4)
    return filters


layer1_filters = build_filters(kernel_size=np.arange(3, 3 + LAYER_TWO_KERNEL_SIZE_NUM * 2, 2),
                               sigma=np.arange(2, LAYER_ONE_KERNEL_SIZE_NUM + 2),
                               lambd=np.arange(2, LAYER_ONE_KERNEL_SIZE_NUM + 2),
                               theta=np.arange(0, np.pi, np.pi / KERNEL_DIRECTION_NUM))
layer2_filters = build_filters(kernel_size=np.arange(3, 3 + LAYER_TWO_KERNEL_SIZE_NUM * 2, 2),
                               sigma=np.arange(2, LAYER_TWO_KERNEL_SIZE_NUM + 2),
                               lambd=np.arange(2, LAYER_TWO_KERNEL_SIZE_NUM + 2),
                               theta=np.arange(0, np.pi, np.pi / KERNEL_DIRECTION_NUM),
                               channel=LAYER_ONE_KERNEL_SIZE_NUM * KERNEL_DIRECTION_NUM)
