import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.models import load_model
import os
import cv2


def buildConfusionMatrix(model_name, x_test, y_test):

    class_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    validation_data_dir = 'data_one'
    model = load_model(model_name)
    img_width = 28
    img_height = 28

    X_test = []
    Y_test = []
    Y_pred = []
    dir_count = -1
    # Get X_test set of data
    # Get Y_test (the labels) of the test set

    dir_count = -1
    for subdir, dirs, files in os.walk(validation_data_dir):
        for file in files:
            image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_width, img_height))
            image = np.array(image).reshape(1,28, 28, 1)
            X_test.append(image)
            Y_test.append(dir_count)
        dir_count+=1

    x_train_cust, x_test_cust, y_train_cust, y_test_cust = train_test_split(
        x_train_cust, y_train_cust, test_size=0.10, random_state=42)
    #
    # X_test = x_test
    # Y_test = y_test


    # Predict Y_Pred using our model
    for image in X_test:
        image = image.reshape(1,img_height, img_width, 1)
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

        fmt = '.1f' if normalize else 'd'
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
    np.set_printoptions(precision=4)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

if __name__ == '__main__':
    # test1.py executed as script
    # do something
    list1 = []
    list2 = []
    buildConfusionMatrix('26_kaggle+personal.h5', list1, list2 )