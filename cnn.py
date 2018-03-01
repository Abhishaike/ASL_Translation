# Import libraries and modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Make a generator that performs data augmentation
train_datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory('Training', target_size=(256, 256), batch_size = 16, class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory('Testing', target_size=(256, 256), batch_size = 16, class_mode = 'binary')


# Define model architecture
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(256,256,3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
# fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# output layer
model.add(Dense(1, activation='softmax'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Fit model
model.fit_generator(train_generator,steps_per_epoch= 10, epochs=50, validation_data=validation_generator, validation_steps=10)
# save the weights of the model
model.save_weights('test.h5')
