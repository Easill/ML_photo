# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 02:01:32 2021

@author: Elias
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

batch_size = 5000
image_size = (256,256)

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory="./projet_python/Training_Data/",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

data_test = tf.keras.preprocessing.image_dataset_from_directory(
    directory="./projet_python/Test_Data",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

x_train = None

for image, label in tfds.as_numpy(data_train):
  print(type(image), type(label), label, len(label))
  x_train = image
  print('------')
  
x_test = None

for image, label in tfds.as_numpy(data_test):
  print(type(image), type(label), label, len(label))
  x_test = image
  print('------')
  
  
  
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# X = df[['c­ol1­','­col­2',­etc.]]
# create df features
# y = df['col']
# create df var to predict
# X_train, X_test, y_train, y_test =
# train_test_split(
#   X,
#   y,
#   test_size=0.3)

# split df in train and test df


# tree = Decisi­onT­ree­Cla­ssi­fier()
# instatiate model
# tree.f­it(­X_t­rain, y_train)
# train/fit the model

# pred = tree.p­red­ict­(X_­test)
# make predic­tions

# print(­cla­ssi­fic­ati­on_­rep­ort­(y_­tes­t,p­red))
# print(­con­fus­ion­_ma­tri­x(y­_te­st,­pred))