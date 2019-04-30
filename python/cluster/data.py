# -*- coding: utf-8 -*-
"""
@Time    : 4/30/19 11:40 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical


class DataSet:
    train_data = pd.read_csv("../data/train_data_gabor.csv", dtype=np.float32)
    test_data = pd.read_csv("../data/test_data_gabor.csv", dtype=np.float32)
    train_labels = to_categorical(train_data["target"].values)
    train_images = train_data.drop("target", axis=1).values
    test_labels = to_categorical(test_data["target"].values)
    test_images = test_data.drop("target", axis=1).values
