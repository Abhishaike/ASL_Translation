import cv2
import PIL

camera_port = 0

frames = 600

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

	
while(True):
	img = get_image()
	im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	im_gray = cv2.blur(im_gray, (5,5))

	cv2.imshow("Gray", im_gray)
	
	img_bw = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)[1]
	canny = cv2.Canny(im_bw, 15, 50)
	
	
	cv2.imshow("Black & white", img_bw)
	
	_, contours,hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	
	print("Contours ", len(contours))
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if(w < 70 or h < 70):
			continue
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.imshow("Canny Stream", canny)
	cv2.imshow("Contours", img)
	
	cv2.waitKey(10)
	
del(camera)	
