# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##CNN
#Block1
#Imports TensorFlow, Keras, and other required libraries for building and training the CNN model.
import os
#import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)
print("Libraries imported successfully")

#Block2
#Specifies the locations of the Angry, Happy, Neutral, Sad, and Surprise image folders.
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
print("Directories created successfully")


#Block3
#Reads the image filenames from the Angry and Happy folders to verify that the images are available.
train_Angry_names = os.listdir(Angry_dir)
print(train_Angry_names[:5])

train_Happy_names = os.listdir(Happy_dir)
print(train_Happy_names[:5])

#Block4
#Creates an ImageDataGenerator and normalizes pixel values from 0–255 to 0–1.
batch_size = 16

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
print("ImageDataGenerator created")

#Block5
#Loads 24,282 images, resizes them to 48×48, converts them to grayscale, and assigns them to the 5 classes.
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'C:\Users\LIKHITHA\Downloads\happy',  # This is the source directory for training images
        target_size=(48, 48),  # All images will be resized to 48 x 48
        batch_size=batch_size,
        color_mode='grayscale',
        
        
        # Specify the classes explicitly
        classes = ['Angry','Happy','Neutral','Sad','Surprise'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
print("Training images loaded")

#Block6
#Builds the Convolutional Neural Network using convolution, pooling, flattening, and dense layers.
target_size=(48,48)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 48*48 with 3 bytes color

     # The first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The second convolution####
    ###tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    ###tf.keras.layers.MaxPooling2D(2,2),
    
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 64 neuron in the fully-connected layer
    tf.keras.layers.Dense(64, activation='relu'),
    
    
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5,activation='softmax')
])
print("CNN model created")

#Block7
#Displays the CNN structure, output shapes, and number of trainable parameters (140,421).
model.summary()

#Block8
#Configures the model using categorical cross-entropy loss and the Adam optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['acc'])#RMSprop(lr=0.001)
print("Model compiled successfully")

#Block9
#Gets the total number of training images: 24,282.
# Total sample count

total_sample=train_generator.n
print("Total samples:", total_sample)

#Block10
#Trains the CNN on the images for 5 epochs and learns to classify the five emotions.
# Training
num_epochs = 5
model.fit(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)

#Block11
#Saves the trained model's architecture into modelGG.json.
# serialize model to JSON
model_json = model.to_json()
with open("modelGG.json", "w") as json_file:
    json_file.write(model_json)
print("Model architecture saved to modelGG.json")

#Block12
#Saves the learned weights into model1GG.weights.h5.
# serialize weights to HDF5
model.save_weights("model1GG.weights.h5")
print("Saved model to disk")



















