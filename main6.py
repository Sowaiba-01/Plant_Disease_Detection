import easygui
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers

data_dir = "C:/Users/SowaibaArshad/PycharmProjects/PlantVillage/PlantVillage-Dataset-master/raw/CNN/Train"
class_names = sorted([item.name for item in pathlib.Path(data_dir).glob('*')])
model_path = "C:/Users/SowaibaArshad/PycharmProjects/DIP_Project/my_model.h5"
model = keras.models.load_model(model_path)

def recogout():
    img_path = easygui.fileopenbox(title="Select Image File")
    if img_path is not None:
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print("This image most likely belongs to {} with a {:.2f} percent confidence."
              .format(class_names[np.argmax(score)], 100 * np.max(score)))
    else:
        print("No image file selected.")

recogout()
