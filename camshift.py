import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


def returnSegmented(img):
    global track_window
    global term_crit
    # Replace with the image you want as your comparison
    roi = cv2.imread('blue.jpg')
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    target = img;
    hsvt = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)
    # calculating object histogram
    # roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    # possible idea: contour the backproject to help meanshift
    dst = cv2.calcBackProject([hsvt], [0], roi_hist, [0, 180], 1)
    # calculated contours based on the back projection
    _,cnts,_ = cv2.findContours(dst.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours from highest to lowest
    cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    # draw the largest contour in white
    cv2.drawContours(dst, cnts,0, (255, 255, 255), 3)
    # draw the rectangle based on the box
    x,y,w,h = cv2.boundingRect(cnts[1])
    cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
    img2 = meanshift(dst, img)
    return img2

def meanshift(dst, img):
    global track_window
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x, y, w, h = track_window
    img2 = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    return img2
def camshift(dst, img):
    global track_window
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
       # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(img,[pts],True, 255,2)
    return img2
def streamSegmented():
    checkColor = False
    # load the image

    #Check if error given on run
    cap = cv2.VideoCapture(0)
    while (cap.isOpened):
        # start timer
        start = time.time()
        _, img = cap.read()
        # img = cv2.imread(args["image"])
        original_img = returnSegmented(img)
        cv2.imshow("Hand contours with bounding box", original_img)
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


if __name__ == "__main__":
    r, h, c, w = 100, 100, 100, 100
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_window = (c, r, w, h)
    streamSegmented()
