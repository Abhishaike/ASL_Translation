import numpy as np
import cv2 as cv
import time

img = cv.imread('images/bh.png',0)
ret,thresh = cv.threshold(img,127,255,0)
im2,contours,hierarchy = cv.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv.moments(cnt)

def doshit():
	clone = img.copy()
	cv.waitKey(1)
	cv.imshow("Window", clone)
	cv.waitKey(1)
	time.sleep(2.0)

doshit()
print( M )