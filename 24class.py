from __future__ import print_function
import keras
# Using thenao input shape is (1, img_height, img_width )
# While tensorflow uses(img_height, img_width, 1 )
# The below setting makes keras use tensorflow format
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Input, BatchNormalization, Conv2D, Lambda, \
    Dense, Dropout, SeparableConv2D
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
import numpy

import cv2
import os

epochs = 6
batch_size = 32
train_data_dir = 'data/train'
validation_data_dir = 'data/test'
img_height = 100
img_width = 100
input_shape = (img_height, img_width, 1)
num_classes = 3

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   fill_mode="nearest",

                                   )
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')

x_train = []
y_train = []
x_test = []
y_test = []
dir_count = -1

# Loading x_train and y_train
for subdir, dirs, files in os.walk(train_data_dir):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        x_train.append(image)
        y_train.append(dir_count)
    dir_count+=1

for subdir, dirs, files in os.walk(validation_data_dir):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        x_test.append(image)
        y_test.append(dir_count)
    dir_count+=1

x_train = numpy.array(x_train)
x_test = numpy.array(x_test)
print(x_train.shape)

InputImg = Input(shape = (100, 100, 3))

ConvModel = Conv2D(32, (3,3), padding = 'same', activation = 'relu', kernel_regularizer = 'l2')(InputImg)
ConvModel = BatchNormalization()(ConvModel)
ConvModel = Conv2D(32, (3,3), padding = 'same', activation = 'relu', kernel_regularizer = 'l2')(ConvModel)
ConvModel = BatchNormalization()(ConvModel)
ConvModel = MaxPooling2D((2,2))(ConvModel)

ConvModel = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_regularizer = 'l2')(ConvModel)
ConvModel = BatchNormalization()(ConvModel)
ConvModel = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_regularizer = 'l2')(ConvModel)
ConvModel = BatchNormalization()(ConvModel)
ConvModel = MaxPooling2D((2,2))(ConvModel)

ConvModel = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_regularizer = 'l2')(ConvModel)
ConvModel = BatchNormalization()(ConvModel)
ConvModel = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_regularizer = 'l2')(ConvModel)
ConvModel = BatchNormalization()(ConvModel)
ConvModel = MaxPooling2D((2,2))(ConvModel)

DenseModel = Dense(512,  activation = 'relu', kernel_regularizer = 'l2')(Flatten()(ConvModel))
DenseModel = BatchNormalization()(DenseModel)
DenseModel = Dropout(.5)(DenseModel)
DenseModel = Dense(256,  activation = 'relu', kernel_regularizer = 'l2')(DenseModel)
DenseModel = BatchNormalization()(DenseModel)
DenseModel = Dropout(.5)(DenseModel)
DenseModel = Dense(128,  activation = 'relu', kernel_regularizer = 'l2')(DenseModel)
DenseModel = BatchNormalization()(DenseModel)
DenseModel = Dropout(.5)(DenseModel)
DenseModel = Dense(num_classes,  activation = 'softmax')(DenseModel)

def poly_decay(epoch):
    maxEpochs = epochs
    baseLR = 1e-3
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=100,
                                              verbose=1,
                                              mode='auto')
saveBestModel = keras.callbacks.ModelCheckpoint('BESTWEIGHTS_FILE',
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='auto')

sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
Model_Complete = Model(inputs = [InputImg], outputs = [DenseModel])
Model_Complete.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

Model_Complete.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks = [saveBestModel, earlyStopping, LearningRateScheduler(poly_decay)])

Model_Complete.save('24class_no_preprocess_longer.h5')