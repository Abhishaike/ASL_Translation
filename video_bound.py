import cv2
import PIL
import cv2.dnn
import numpy as np

camera_port = 0

camera = cv2.VideoCapture(camera_port)

def get_image():
	retval, im = camera.read()
	return im

def draw_contour(contour, img, threshW, threshH):
	x,y,w,h = cv2.boundingRect(cnt)
	if(w < threshW or h < threshH):
		return False
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
	return True

def rect(img,(x,y,w,h)):
	cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 3)	

while(True):
	img = get_image()
	im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	im_gray = cv2.GaussianBlur(im_gray, (7,7), 0)

	cv2.imshow("Gray", im_gray)
	
	img_bw = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)[1]
	
	canny = cv2.Canny(im_bw, 100, 200)
		
	cv2.imshow("Black & white", img_bw)
	
	_, contours,hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
	#	x,y,w,h = cv2.boundingRect(cnt)
	#	if(w < 50 or h < 50):
	#		continue
		epsilon = 0.01*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		#cv2.polylines(img, np.int32(approx), True, thickness=4, color=(0,255,0))
		hull = cv2.convexHull(approx)
		cv2.polylines(img, np.int32(hull), True, thickness=4, color=(255,0,0))
		#rect(img, cv2.boundingRect(np.int32(hull)))



	#	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.imshow("Canny Stream", canny)
	cv2.imshow("Contours", img)
	
	cv2.waitKey(5)
	
del(camera)	
