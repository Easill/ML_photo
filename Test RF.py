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
import pandas as pd
import tensorflow_datasets as tfds

# IMPORT MODELLING LIBRARIES
from sklearn.model_selection import train_test_split

# libraries for decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

# libraries for random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

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

for image, label_train in tfds.as_numpy(data_train):
  print(type(image), type(label_train), label_train, len(label_train))
  x_train = image
  print('------')
  
x_test = None

for image, label_test in tfds.as_numpy(data_test):
  print(type(image), type(label_test), label_test, len(label_test))
  x_test = image
  print('------')
  
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

xtrain = np.reshape(x_train, (1107,196608))
xtest = np.reshape(x_test, (3799, 196608))

df_train = pd.DataFrame(data=xtrain)
df_test = pd.DataFrame(data=xtest)

tentative_train = pd.DataFrame(label_train)

value = range(1107)
print(value)

Label = []

for i in value:
        if tentative_train.iloc[i,0]==1:
            Label=Label+[0]
        elif tentative_train.iloc[i,1]==1:
            Label=Label+[1]
        else :
            Label=Label+[2]



df_train['Label']=Label

val = range(196607)
cols = [0]
for i in val:
    cols = cols + [i]


df_train.head()

tan = df_train[cols]

X_train = tan
# create df features
y_train = df_train['Label']
# create df var to predict

# split df in train and test df

rfc = RandomForestClassifier()
n_estimators=200
# instatiate model
rfc.fit(X_train,y_train)

# train/fit the model

tentative_test = pd.DataFrame(label_test)

value = range(3799)
print(value)

Label = []

for i in value:
        if tentative_test.iloc[i,0]==1:
            Label=Label+[0]
        elif tentative_test.iloc[i,1]==1:
            Label=Label+[1]
        else :
            Label=Label+[2]
            
df_test['Label']=Label

val = range(196607)
cols = [0]
for i in val:
    cols = cols + [i]

ten = df_test[cols]

X_test = ten
y_test = df_test['Label']

rfc_pred = rfc.predict(X_test)

# EVAUATE MODEL

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
