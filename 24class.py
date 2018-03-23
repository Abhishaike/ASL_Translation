from __future__ import print_function
import keras

# Using thenao input shape is (1, img_height, img_width ) 
# While tensorflow uses(img_height, img_width, 1 )
# The below setting makes keras use tensorflow format
from keras import backend as K
K.set_image_dim_ordering('tf')

import cv2
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
from PIL import Image
import numpy

batch_size = 32
epochs = 15
train_data_dir = 'data/train'
validation_data_dir = 'data/test'

img_height = 100
img_width = 100
input_shape = (img_height, img_width, 1)
num_classes = 24

# This seems to improve real-time accuracy
# Currently if this is not done, the loss function never goes below 1.4
def preprocess(image):
    open_cv_image = numpy.array(image)
    blur = cv2.GaussianBlur(open_cv_image,(5,5),2)   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, processed_image = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    im_pil = Image.fromarray(processed_image)
    return im_pil

# Room for improvement here. Maybe width, height, and zoom scale can be added. 
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   fill_mode="nearest",
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   preprocessing_function=preprocess)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess)

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

# Room to make the model better. Need to add batch normalization.
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    )

model.save('24class_model.h5')
