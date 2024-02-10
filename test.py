import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Load the model
model = tf.keras.models.load_model('Models/drawings_model.keras')

# Load the dataset
train_generator = ImageDataGenerator().flow_from_directory('Dataset', target_size=(256, 256))

# Predict the class of an image
def predict_image(filename):
    img = load_img(filename, target_size=(256, 256)) 

    img_to_show = ImageTk.PhotoImage(img)

    img = img_to_array(img) / 255.

    img = np.expand_dims(img, axis=0)

    probs = model.predict(img)

    class_index = np.argmax(probs)

    class_label = list(train_generator.class_indices.keys())[class_index]

    return img_to_show, class_label

# Displaying the images with result using tkinter

root = tk.Tk()

images = []

for i, filename in enumerate(['TestImages/test1.png', 'TestImages/test2.png', 'TestImages/test3.png']):
    img, label = predict_image(filename)

    images.append(img)

    img_label = tk.Label(root, image=img, text=f"This is a '{label}'.", compound=tk.TOP)

    img_label.grid(row=0, column=i)

root.mainloop()