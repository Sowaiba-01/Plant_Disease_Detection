import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tkinter as tk
from tkinter import filedialog

data_dir = "C:/Users/SowaibaArshad/PycharmProjects/PlantVillage/PlantVillage-Dataset-master/raw/CNN/Train"
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "training", seed = 123,
    image_size = (224, 224), batch_size = batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "validation", seed = 123,
    image_size = (224, 224), batch_size = batch_size)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
num_class = 8
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (224, 224, 3)),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(num_class)
])

noepochs = 7
model.compile(optimizer = 'adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'])
mymodel = model.fit(train_ds, validation_data = val_ds, epochs = noepochs)
acc = mymodel.history['accuracy']
val_acc = mymodel.history['val_accuracy']
loss = mymodel.history['loss']
val_loss = mymodel.history['val_loss']
epochs_range = range(noepochs)
total_epochs = len(acc)
total_accuracy = acc[total_epochs - 1]

print("Total Accuracy: {:.2%}".format(total_accuracy))
