
import cv2
import numpy as np
import argparse
# construct the arguements
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())
# read image
img = cv2.imread(args["image"],1)
# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# added a smoothing filter that preserved the edges
gray = cv2.bilateralFilter(gray, 11, 17, 17)
# added a canny filter to detect edges
edged = cv2.Canny(gray, 30, 200)
# calculated contours
_,cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# sort contours from highest to lowest
cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# # loop over our contours
# for c in cnts:
# 	# approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.05 * peri, True)
#     print(approx)
# 	# determines the contour to draw on
#     if len(approx) == 2:
# 	    screenCnt = approx
# 	    break
# draw the contours onto the image
cv2.drawContours(img, cnts, 0, (0, 255, 0), 3)
# make a bounding box to the hand
x,y,w,h = cv2.boundingRect(cnts[0])
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Hand contours with bounding box", img)
cv2.imwrite('bounding_box.png',img)
# close the image window when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
quit()
