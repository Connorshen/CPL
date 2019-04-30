# -*- coding: utf-8 -*-
"""
@Time    : 4/30/19 10:41 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""

import pickle


class Parameter:
    middle_neuron_num = 20000
    input_neuron_num = pickle.load(open("../data/data_info.pickle", "rb"))["train_images_shape"][1]
    output_neuron_num = 10
    cluster_neuron_num = 10
    input_weight_density = 0.01
