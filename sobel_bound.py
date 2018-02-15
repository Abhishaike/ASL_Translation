import cv2
import numpy as np
import sys

imgloc = sys.argv[1]

img = cv2.imread(imgloc)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_bw = cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY)[1]
_, contours,hierarchy = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print len(contours)

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
	hull = cv2.convexHull(cnt)
	for i in cnt:
		cv2.line(img, (i[0][0] - 1, i[0][1] - 1), (i[0][0] + 1, i[0][1] + 1),(0, 0, 0), 1)
		continue
	cv2.polylines(img, hull, True, (0, 255, 255), 3)
	
#x,y,w,h = cv2.boundingRect(cnt)
#img_gray = cv2.rectangle(img_gray,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()