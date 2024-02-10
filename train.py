import tensorflow as tf
from model import create_model
import matplotlib.pyplot as plt

# Create an ImageDataGenerator object with data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training images
train_generator = datagen.flow_from_directory(
    'Dataset/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation images
validation_generator = datagen.flow_from_directory(
    'Dataset/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Create the model
model = create_model()

# Add dropout layers for regularization
model.add(tf.keras.layers.Dropout(0.5))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=[early_stopping]
)

# Save the model
model.save('Models/drawings_model.keras')
