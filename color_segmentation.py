# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
ap.add_argument("-k", "--kvalue", help="num segmentations")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

Z=image.reshape((-1,3))
Z=np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K=int(args["kvalue"])
attempts=10
ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
print(center)
res=center[label.flatten()]
res2=res.reshape((image.shape))
# for i in range(len(res2)):
#     i-=1
#     for y in range(len(res2[i])):
#         y-=1
# #        print(y)
#         if(np.array_equal(res2[i][y],np.array([108,72,88]))):
#
#             res2[i][y] = np.array([255,255,255])
#         else:
#             res2[i][y]=np.array([0,0,0])
gray = res2
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# c = max(cnts, key=cv2.contourArea)
#
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])

cv2.imshow('res2',gray)
cv2.waitKey(0)

cv2.destroyAllWindows();
quit();