import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.models import load_model
import os
import cv2

class_names = ['a','b','c']
validation_data_dir = 'data/train'
model = load_model('3_small_data_aug.h5')

X_test = []
Y_test = []
Y_pred = []
dir_count = -1
# Get X_test set of data
# Get Y_test (the labels) of the test set
for subdir, dirs, files in os.walk(validation_data_dir):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        shaped = image.reshape(1, 100, 100, 3) #reshaping for nn
        X_test.append(shaped)
        Y_test.append(dir_count)
    dir_count+=1

# Predict Y_Pred using our model
for image in X_test:
    result = model.predict(image, batch_size=1)[0] # Predict
    Y_pred.append(np.argmax(result))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()