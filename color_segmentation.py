# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time
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
    _,img = cap.read()
    #img = cv2.imread(args["image"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[::2, ::2]
    # save a copy
    original_image = img.copy()
    segments_fz = felzenszwalb(img, scale=1, sigma=0.9, min_size=1000)
    # segment color is the color of the hand
    segmentColor = None
    # convert to hsv for seperation
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # set limits of blue
    rangeMask = cv2.inRange(hsv, lower_blue, upper_blue)
    # apply color mask
    img = cv2.bitwise_and(img,img,mask = rangeMask)
    # threshold the image to remove random blue noise from the image
    _,thresh = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)
    # finds the segment color when there are threshold values
    indicies = np.where(thresh == 1)
    segmentColor = segments_fz[indicies]
    # vectorized approach
    indicies = np.where(segments_fz == segmentColor)
    img[indicies] = [255,255,255]
    img[not indicies] = [0,0,0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # calculated contours
    _,cnts,_ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours from highest to lowest
    cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    # draw the first contour
    cv2.drawContours(original_image, cnts,0, (0, 255, 0), 3)
    # assumes that the hand is the biggest contour
    # make a bounding box to the hand
    # later make the box size constant so that HOGS works well
    x,y,w,h = cv2.boundingRect(cnts[0])
    cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Hand contours with bounding box",original_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # stop the clock
    end = time.time()
    print(end - start)
    #close the image window when a key is pressed
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # quit
cap.release()
cv2.destroyAllWindows
