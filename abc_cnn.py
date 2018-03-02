from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

batch_size = 32
epochs = 8
train_data_dir = 'databinary/train'
validation_data_dir = 'databinary/test'

img_height = 200
img_width = 200
input_shape = (img_height, img_width, 1)
num_classes = 3

#data augmentation, fit it to training data. NOT testing data. 
train_datagen = ImageDataGenerator(width_shift_range=0.1,
rescale=1./255, 
height_shift_range=0.1, horizontal_flip=True, vertical_flip=True,
                             fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse')

# Network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

def poly_decay(epoch):
    maxEpochs = 100
    baseLR = 1e-3
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit_generator(generator=train_generator,
          epochs=epochs,
          verbose=1,
          validation_data=validation_generator,
          )

model.save('abc_binary_model.h5')
