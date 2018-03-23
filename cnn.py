# Import libraries and modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization, Average
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD

# Make a generator that performs data augmentation
train_datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,fill_mode='nearest',zca_whitening=True)
test_datagen = ImageDataGenerator(rescale = 1./255,zca_whitening=True)
train_generator = train_datagen.flow_from_directory('Training', target_size=(128, 128), batch_size = 100,color_mode = 'grayscale')
validation_generator = test_datagen.flow_from_directory('Testing', target_size=(128, 128), batch_size = 100,color_mode = 'grayscale')


# Define model architecture
model = Sequential()

model.add(Convolution2D(100, (3, 3), activation='relu', input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(Convolution2D(75, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(50, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(35, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(10, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(5, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
# fully connected layers
model.add(Dense(64, activationas='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# output layer
model.add(Dense(24, activation='softmax'))
# Compile model
# setup stochastic gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# Fit model
model.fit_generator(train_generator,steps_per_epoch= 10, epochs= 100, validation_data=validation_generator, validation_steps=10)
# save the weights of the model
model.save_weights('10_classes_test_with_batch_withGrayScale.h5')

def ensemble(model, model_input):
    output = [models.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input,y,name = 'ensemble')
    return model
