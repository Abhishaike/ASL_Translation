import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time
from scipy.stats import iqr
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help="path to the image")
# ap.add_argument("-k", "--kvalue", help="num segmentations")
# args = vars(ap.parse_args())

# load the image
cap = cv2.VideoCapture(0)
# preallocate memory
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])


while(cap.isOpened):
    # start timer
    start = time.time()
    currentTime = time.time()
    _,img = cap.read()
    currentTime = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # finds the faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # gets the coordinates and size of the rectangle that has the face
    x = faces[0,0]
    y = faces[0,1]
    w = faces[0,2]
    h = faces[0,3]
    roi_color = img[y:y + h, x:x + w]
    hsv_img = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # isolate each color channel
    h = hsv_img[:,:,0]
    s = hsv_img[:,:,1]
    v = hsv_img[:,:,2]
    # calculate iqr and median of each channel
    # percentile = np.percentile(h,25)
    # iqr_array = [iqr(h),iqr(s),iqr(v)]
    # median_array = [np.median(h),np.median(s),np.median(v)]
    # calculate lower and upper bound of the skin color by calculating 25 and 75 percentile of each color channel
    lower_bound = [np.percentile(h,25),np.percentile(s,25),np.percentile(v,25)]
    upper_bound = [np.percentile(h,75),np.percentile(s,75),np.percentile(v,75)]
    print(lower_bound)
    print(upper_bound)
    cv2.imshow("Facial Recognition", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # stop the clock
    end = time.time()
    # print(end - start)
    # close the image window when a key is pressed
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # quit

cap.release()
cv2.destroyAllWindows
