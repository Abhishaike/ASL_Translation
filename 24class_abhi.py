from __future__ import print_function
import keras
# Using thenao input shape is (1, img_height, img_width )
# While tensorflow uses(img_height, img_width, 1 )
# The below setting makes keras use tensorflow format
from keras import backend as K
K.set_image_dim_ordering('tf')
import cv2
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
from PIL import Image
import numpy

epochs = 100
batch_size = 32
train_data_dir = 'data/train'
validation_data_dir = 'data/test'
img_height = 100
img_width = 100
input_shape = (img_height, img_width, 1)
num_classes = 3
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

def poly_decay(epoch):
    maxEpochs = epochs
    baseLR = 1e-3
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha


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

Model_Complete.fit_generator(generator=train_generator,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks = [saveBestModel, earlyStopping, LearningRateScheduler(poly_decay)])

Model_Complete.save('24class_no_preprocess_longer.h5')