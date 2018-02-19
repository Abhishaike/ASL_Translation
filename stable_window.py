import cv2
import numpy as np

camera_port = 0

camera = cv2.VideoCapture(camera_port)
fgbg = cv2.createBackgroundSubtractorMOG2()

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_image():
	retval, im = camera.read()
	return im

while(True):
	img = get_image()

	img = cv2.resize(img, (0,0), fx=2, fy=2)
	fgmask = fgbg.apply(img)

	#cv2.rectangle(img, (400, 400), (600,600), (0,0,255), 5)


	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(fgmask, 127, 255, 0)
	_,contours,hierarchy = cv2.findContours(thresh, 2, 1)

	img = cv2.GaussianBlur(img, (3,3), 200)

	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)

	for i in range(1, 5):
		cv2.drawContours(img, contours, i, (0,255,0), 4, 8)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)

		if(w * h < 20000):
			continue

		cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 4)

	cv2.imshow("delta", fgmask)

	cv2.imshow('img',img)

	cv2.waitKey(1)
	
del(camera)	
