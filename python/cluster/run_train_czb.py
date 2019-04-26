# -*- coding: utf-8 -*-
"""
@Time    : 4/26/19 8:31 PM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

train_data = pd.read_csv("../data/train_data_czb.csv", header=None)
train_labels = to_categorical(train_data[0].values)
train_images = train_data.drop(0, axis=1)

test_data = pd.read_csv("../data/test_data_czb.csv", header=None)
test_labels = to_categorical(test_data[0].values)
test_images = test_data.drop(0, axis=1)

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
