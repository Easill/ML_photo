# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:23:07 2021

@author: pierr
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# 0. Problématique : Prédire la classe d'une photo 
#  - d'un arbre ;
#  - d'une fleur ;
#  - d'un autre "objet".

print("\n========================")
print("1. Chargement des images")
print("========================")

# Création d'un jeu de données d'images à partir d'un répertoire
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="./photos",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

print("==========")
print(type(dataset))
print("==========")
print(dataset)

print("\n============================")
print("2. Prétrairement des données")
print("============================")


# For demonstration, iterate over the batches yielded by the dataset.
for data, labels in dataset:
   print(data.shape)  # (64,)
   print(data.dtype)  # string
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
print("==========")

# from tensorflow.keras.layers import Rescaling
# cropper = keras.layers.CenterCrop(height =150, width = 150)
# output_data = cropper(dataset)
# rescale = Rescaling(scale=1/256)(dataset)
dense = keras.layers.Dense(units = 10)
inputs = keras.Input(shape=(256,256,1))

data_train = np.random.randint(0, 256, size=(32, 256, 256, 1)).astype("float32")
print("==========")

print("\n=========================================")
print("3. Construction de la structure du modèle")
print("=========================================")

# Center-crop images to 150x150
# x = layers.CenterCrop(height=150, width=150)(inputs)
# Rescale images to [0, 1]
x = layers.Rescaling(scale=1.0 / 255)(inputs)
print(x)
# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 3
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

print("\n=========================")
print("3. Entraînement du modèle")
print("=========================")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)



history = model.fit(dataset, epochs=10)
# print(history.history["loss"])

# fonction de perte du modèle à chaque "epoch"
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


















