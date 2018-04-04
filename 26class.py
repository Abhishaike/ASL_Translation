from __future__ import print_function
import keras
from keras.preprocessing import image as keras_image
from keras import backend as K
from keras.layers import MaxPooling2D, Flatten, Input, BatchNormalization, Conv2D, \
    Dense, Dropout, SeparableConv2D, Activation, Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
from sklearn.utils import shuffle
import csv
import numpy
import cv2
import os
from sklearn.model_selection import train_test_split

import confusion_matrix # Optional code for generating confusion matrix at the end

K.set_image_dim_ordering('tf')

epochs = 15
batch_size = 32
train_data_dir = 'data_one'
# validation_data_dir = 'data/test'
img_height = 28
img_width = 28
input_shape = (img_height, img_width, 1)
num_classes = 26
model_name = '26_aug_kag+cust.h5'

x_train = []
y_train = []
# Loading x_train and y_train
# 28 x 28
with open("sign_mnist_train.csv", "r") as infile:
    reader = csv.reader(infile)
    headers = next(reader)[1:]
    for row in reader:
        y_train.append(int(row[0]))
        image = []
        image_row = []
        pix_row = 0
        for pix in row[1:]:
            image_row.append(int(pix))
            if pix_row is 27:
                image.append(image_row)
                pix_row = 0
                image_row = []
            else:
                pix_row+=1

        image = numpy.array(image).reshape(28,28,1)
        x_train.append(image)

x_test = []
y_test = []
# Loading x_train and y_train
# 28 x 28
with open("sign_mnist_test.csv", "r") as infile:
    reader = csv.reader(infile)
    headers = next(reader)[1:]
    for row in reader:
        y_test.append(int(row[0]))
        image = []
        image_row = []
        pix_row = 0
        for pix in row[1:]:
            image_row.append(int(pix))
            if pix_row is 27:
                image.append(image_row)
                pix_row = 0
                image_row = []
            else:
                pix_row+=1

        image = numpy.array(image).reshape(28,28,1)
        x_test.append(image)

# Adding my own data_set
x_train_cust = []
y_train_cust = []
x_test_cust = []
y_test_cust = []
dir_count = -1
for subdir, dirs, files in os.walk(train_data_dir):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (img_width, img_height))
        image = numpy.array(image).reshape(28, 28, 1)
        x_train_cust.append(image)
        y_train_cust.append(dir_count)
    dir_count+=1

x_train_cust, x_test_cust, y_train_cust, y_test_cust = train_test_split(
            x_train_cust, y_train_cust, test_size=0.10, random_state=42)

# Add the custom data to the kaggle dataset
x_train.extend(x_train_cust)
y_train.extend(y_train_cust)
x_test.extend(x_test_cust)
y_test.extend(y_test_cust)

# Image preproccesing
for i, img in enumerate(x_train):
    x_train[i] = keras_image.random_rotation(x_train[i], rg=10, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0)
    x_train[i] = keras_image.random_shift(x_train[i], wrg=0.1, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0)
    x_train[i] = keras_image.random_zoom(x_train[i], zoom_range=(0.85,1.15), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0)

x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
x_test = numpy.array(x_test)
y_test = numpy.array(y_test)

x_train, y_train = shuffle(x_train, y_train, random_state=0)

InputImg = Input(shape = (img_height, img_width, 1))

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
                   validation_data=(x_test,y_test),
                   batch_size=batch_size,
                    callbacks = [saveBestModel, earlyStopping, LearningRateScheduler(poly_decay)])

Model_Complete.save(model_name)

confusion_matrix.buildConfusionMatrix(model_name, x_test, y_test)