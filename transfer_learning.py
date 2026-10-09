# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:42:56 2026

@author: LIKHITHA
"""
##CNN

########transfer learning
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------------------------
# STEP 1: Directories (same as your original code)
# -----------------------------------------------------
#Directory with angry images
Angry_dir= os.path.join(r'C:\Users\LIKHITHA\Downloads\happy\angry')

# Directory with Happy images
Happy_dir= os.path.join(r'C:\Users\LIKHITHA\Downloads\happy\happy')

# Directory with Neutral images
Neutral_dir = os.path.join(r'C:\Users\LIKHITHA\Downloads\happy\neutral')

# Directory with Sad images
Sad_dir = os.path.join(r'C:\Users\LIKHITHA\Downloads\happy\sad')

# Directory with Surprise images
Surprise_dir = os.path.join(r'C:\Users\LIKHITHA\Downloads\happy\surprise')

train_Angry_names = os.listdir(Angry_dir)
print(train_Angry_names[:5])

train_Happy_names = os.listdir(Happy_dir)
print(train_Happy_names[:5])

batch_size = 16

# -----------------------------------------------------
# STEP 2: Data generators
# -----------------------------------------------------
# Pretrained models like MobileNetV2 expect 3-channel (RGB) input and
# their own specific pixel scaling, so we use preprocess_input instead
# of a plain rescale=1./255.
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# target_size is increased to 96x96 because MobileNetV2 needs a
# minimum input size larger than 48x48 to work well.
target_size = (96, 96)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\LIKHITHA\Downloads\happy',
    target_size=target_size,
    batch_size=batch_size,
    # color_mode is 'rgb' here (not 'grayscale'). Keras will
    # automatically duplicate the single channel into 3 channels,
    # since MobileNetV2 was trained on 3-channel ImageNet images.
    color_mode='rgb',
    classes=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    class_mode='categorical'
)

# -----------------------------------------------------
# STEP 3: Load the pretrained base model
# -----------------------------------------------------
base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,     # remove ImageNet's original 1000-class head
    weights='imagenet'
)

# Freeze the pretrained layers so their weights don't change
# during the first phase of training.
base_model.trainable = False
print("MobileNetV2 loaded successfully")
# -----------------------------------------------------
# STEP 4: Build the full model (pretrained base + your own head)
# -----------------------------------------------------
inputs = tf.keras.layers.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

# 5 output neurons for 5 classes with the softmax activation
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.models.Model(inputs, outputs)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['acc']
)
print("Model compiled successfully")
# -----------------------------------------------------
# STEP 5: Train the new head (base model stays frozen)
# -----------------------------------------------------
total_sample = train_generator.n
num_epochs = 2

model.fit(
    train_generator,
    steps_per_epoch=int(total_sample / batch_size),
    epochs=num_epochs,
    verbose=1
)