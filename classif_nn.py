# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:23:07 2021

@author: pierr
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

image_size=(256, 256)
batch_size=32

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(.5,.2)
    ]
)

# callbacks = [
#     keras.callbacks.TensorBoard(log_dir='./logs')
# ] #Pour garder en mémoire ce qu'il s'est passé pendant le training et la validation
#PS : je n'ai pas encore réussi à l'implémenter dans ce que je voulais...
def get_compiled_model():
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        )
    return model

# 0. Problématique : Prédire la classe d'une photo 
#  - d'un arbre ;
#  - d'une fleur ;
#  - d'un autre "objet".

print("\n========================")
print("1. Chargement des images")
print("========================")

# Création d'un jeu de données d'images à partir d'un répertoire
data_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory="./projet_python",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    interpolation="bilinear",
    validation_split=0.2,
    subset="training",
    seed=123,
    follow_links=False,
    crop_to_aspect_ratio=False
)

data_test = tf.keras.preprocessing.image_dataset_from_directory(
    directory="./projet_python",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    interpolation="bilinear",
    validation_split=0.2,
    subset="validation",
    seed=123,
    follow_links=False,
    crop_to_aspect_ratio=False
)


data_train = data_train.prefetch(buffer_size=32)
data_test = data_test.prefetch(buffer_size=32)

print("==========")
print(type(data_train))
print("==========")
print(data_train)


print("\n============================")
print("Affichage des 9 premières photos du jeu de données train")
print("============================")

plt.figure(figsize=(10, 10))
for images, labels in data_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

print("\n============================")
print("Affichage d'une photo sur laquelle on applique de la data augmentation")
print("============================")


plt.figure(figsize=(10, 10))
for images, _ in data_train.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


print("\n============================")
print("2. Prétraitement des données")
print("============================")


# For demonstration, iterate over the batches yielded by the dataset.
for data, labels in data_train:
   print(data.shape)  # (64,)
   print(data.dtype)  # string
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
print("==========")

# from tensorflow.keras.layers import Rescaling
# cropper = keras.layers.CenterCrop(height =150, width = 150)
# output_data = cropper(dataset)
# rescale = Rescaling(scale=1/256)(dataset)
# dense = keras.layers.Dense(units = 3)

inputs = keras.Input(shape=(256,256,3))


# data_train = np.random.randint(0, 256, size=(32, 256, 256, 1)).astype("float32")
print("==========")

print("\n=========================================")
print("3. Construction de la structure du modèle")
print("=========================================")

# Center-crop images to 150x150
# x = layers.CenterCrop(height=150, width=150)(inputs)
# Rescale images to [0, 1]

x = layers.Rescaling(scale=1.0 / 255)(inputs)
x = data_augmentation(x)
print(x)
# Apply some convolution and pooling layers
# x = layers.Dense(10, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu")(x)
x = layers.Dropout(0.3, seed = 123)(x)
# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 3
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

print("\n=========================")
print("4. Entraînement du modèle")
print("=========================")


model=get_compiled_model()
# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy"
# )

history=model.fit(data_train, epochs=20,validation_data=data_test)
print("=========================")
# print("Evaluate")
# result=model.evaluate(data_test)
print("=========================")
predictions = model.predict(data_test)
print(predictions.shape)
# print(history.history["loss"])



# fonctions de perte du modèle
plt.plot(history.history["loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(history.history["val_loss"])
plt.show()

# accuracies du modèle
plt.plot(history.history["accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(history.history["val_accuracy"])
plt.show()


print("\n=======================")
print("5. Validation du modèle")
print("=======================")


# Get the data as Numpy arrays
# (x_train, y_train), (x_test, y_test) = dataset

# val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
# history = model.fit(dataset, epochs=1, validation_data=val_dataset)











