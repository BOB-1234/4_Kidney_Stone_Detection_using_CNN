# Importing Libraries to train the machine learning model
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator

# Setting the Path for the dataset, based on where you saved the data if in same folder just call the name if not 
# use the directory of the path
path = 'Dataset'

# Loading the images
# use directory path if not located in same folder as your code
train_data_dir = os.path.join(path, 'Train')
test_data_dir = os.path.join(path, 'Test')

# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training the Model
# setting the image sizes as 150,150 and batch size of 32 using the binary class for processing model
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(150,150), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(150,150), batch_size=32, class_mode='binary', shuffle=False)

# Creating the ML model
model = Sequential()

# Adding convolutional layers using relu activation function with a max pooling layer
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
# same as the previous creation of another convolutional layer with max pooling and use of relu activation function
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# converting the feature to a 1 dimensional feature vector
model.add(Flatten())  
model.add(Dense(64)) # adding a dense layer
model.add(Activation('relu')) # using activation function as relu
model.add(Dropout(0.5)) # dropout added on
model.add(Dense(1)) # second dense layer added 
model.add(Activation('sigmoid')) # adding sigmoid as an activation function for processing the model

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the Model
# set the number of epochs
model.fit(train_generator, steps_per_epoch=32, epochs=3, validation_data=test_generator, validation_steps=32)

# Saving the Model as an h5 file
model.save('kidney_stone_model.h5')
