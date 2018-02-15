
import cv2
import numpy as np
img = cv2.imread('evan0.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
_,cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) > 1:
		screenCnt = approx
		break
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Game Boy Screen", img)
#cv2.imshow('img',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
quit()
