# -*- coding: utf-8 -*-
"""
@Time    : 4/25/19 8:56 PM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

"""测试下gabor滤波之后的特征向量是否能通过全连接的形式训练得到较好的结果"""

train_data = pd.read_csv("../data/train_data_gabor.csv")
train_labels = to_categorical(train_data["target"].values)
train_images = train_data.drop("target", axis=1)

test_data = pd.read_csv("../data/test_data_gabor.csv")
test_labels = to_categorical(test_data["target"].values)
test_images = test_data.drop("target", axis=1)

model = Sequential()
model.add(Dense(
    units=100,
    batch_input_shape=(None, train_images.shape[1]),
    activation="relu"
))
model.add(Dropout(0.2))
model.add(Dense(
    units=10,
    activation="softmax"
))
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=32)
loss, accuracy = model.evaluate(test_images, test_labels)
print(loss)
print(accuracy)
