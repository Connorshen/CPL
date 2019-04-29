# -*- coding: utf-8 -*-
"""
@Time    : 4/26/19 10:07 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import pandas as pd
import numpy as np


class DataSet:
    TYPE_TEST = "type_test"
    TYPE_TRAIN = "type_train"
    train_data = pd.read_csv("../data/train_data.csv", dtype=np.float32)
    test_data = pd.read_csv("../data/test_data.csv", dtype=np.float32)
    train_labels = train_data["target"].values
    train_images = train_data.drop("target", axis=1).values.reshape(-1, 28, 28, 1) / 255
    test_labels = test_data["target"].values
    test_images = test_data.drop("target", axis=1).values.reshape(-1, 28, 28, 1) / 255
    test_index = 0
    train_index = 0
    train_data_len = len(train_data)
    test_data_len = len(test_data)

    def train_next_batch(self, batch_num):
        if self.train_data_len % batch_num != 0:
            raise BatchNumError("batch长度必须能被数据集长度整除")
        labels = self.train_labels[self.train_index:batch_num + self.train_index]
        images = self.train_images[self.train_index:batch_num + self.train_index]
        self.train_index += batch_num
        return images, labels

    def test_next_batch(self, batch_num):
        if self.test_data_len % batch_num != 0:
            raise BatchNumError("batch长度必须能被数据集长度整除")
        labels = self.test_labels[self.test_index:batch_num + self.test_index]
        images = self.test_images[self.test_index:batch_num + self.test_index]
        self.test_index += batch_num
        return images, labels


class BatchNumError(Exception):
    pass
