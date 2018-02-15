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
image = cv2.GaussianBlur(image, (11, 11), 2)
Z=image.reshape((-1,3))
Z=np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, .1)
K=int(args["kvalue"])
attempts=10
ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
print(center)
res=center[label.flatten()]
res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
res2=res.reshape((image.shape))

#uncomment if you want to binarize

for i in range(len(res2)):
    i-=1
    for y in range(len(res2[i])):
        y-=1
#        print(y)
        #if(np.array_equal(res2[i][y],np.array([10,23,241]))): #MUST CHANGE THIS TO PURPLE COLOR
        if(abs(res2[i][y][0] - 10) < 25 and abs(res2[i][y][1] - 23) < 25 and abs(res2[i][y][2] - 241)<25):
            res2[i][y] = np.array([255,255,255])
        else:
            res2[i][y]=np.array([0,0,0])
gray = res2
gray = cv2.GaussianBlur(gray, (5, 5), 0)


#don't fuck with this unless you want to do some contour shit
#threshold the image, then perform a series of erosions +
#dilations to remove any small regions of noise
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

# denoise the image
initial_kernel = np.ones((12,12),np.uint8)
closer_kernel = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, initial_kernel)
#cv2.imwrite('binary_output_withMorphology.png',gray)
cv2.imshow('res2',gray)
cv2.waitKey(0)

cv2.destroyAllWindows();
quit();
