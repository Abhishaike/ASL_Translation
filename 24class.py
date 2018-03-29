from __future__ import print_function
import keras
from keras import backend as K
from keras.layers import MaxPooling2D, Flatten, Input, BatchNormalization, Conv2D, \
    Dense, Dropout, SeparableConv2D, Activation, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm

import numpy
import cv2
import os

K.set_image_dim_ordering('tf')

epochs = 10
batch_size = 32
train_data_dir = 'data/train'
validation_data_dir = 'data/test'
img_height = 100
img_width = 100
input_shape = (img_height, img_width, 1)
num_classes = 3

x_train = []
y_train = []
x_test = []
y_test = []

# Loading x_train and y_train
dir_count = -1
for subdir, dirs, files in os.walk(train_data_dir):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        x_train.append(image)
        y_train.append(dir_count)
    dir_count+=1

dir_count = -1
for subdir, dirs, files in os.walk(validation_data_dir):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        x_test.append(image)
        y_test.append(dir_count)
    dir_count+=1

x_train = numpy.array(x_train)
x_test = numpy.array(x_test)

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

Model_Complete.save('3class_glove.h5')