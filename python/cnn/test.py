# -*- coding: utf-8 -*-
"""
@Time    : 4/19/19 10:41 AM
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(
    1024,
    batch_input_shape=(None, 12544),
    activation="relu",

))
model.add(Dense(10, activation="softmax"))
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
