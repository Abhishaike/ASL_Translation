# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
ap.add_argument("-k", "--kvalue", help="num segmentations")
args = vars(ap.parse_args())

# load the image
img = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#image = cv2.GaussianBlur(image, (101,101), 0)


img = img[::2, ::2]
# save a copy
original_image = img
#print(img)
segments_fz = felzenszwalb(img, scale=1, sigma=0.9, min_size=1000)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,
                 subplot_kw={'adjustable': 'box-forced'})
# returns the index of unique segments in a flattened array
_,index_Uniques = np.unique(segments_fz,return_index = True)
# flattens the original image array so that it can be compared with the uniques
flat_image = np.ravel(img)
# segment color is the color of the hand
segmentColor = None
has_color = False
# convert to hsv for seperation
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# set limits of blue
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
rangeMask = cv2.inRange(hsv, lower_blue, upper_blue)
# apply color mask
img = cv2.bitwise_and(img,img,mask = rangeMask)
# convert to gray scale
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# threshold the image to remove random blue noise from the image
_,thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
#cv2.imwrite("test.3.png",thresh)
for i in range(len(img)):
    for j in range(len(img[i])):
        # chooses the segment that has the color of the hand
        #print(img[i,j])
        if(not thresh[i,j] == 0):
            print(img[i,j])
            segmentColor = segments_fz[i,j]
            has_color = True
            break
    # break out of the first for loop
    if(has_color):
        break
for i in range(len(segments_fz)):
    for y in range(len(segments_fz[i])):
        if(segments_fz[i][y]==segmentColor):
            segments_fz[i][y] = 1
            #print(img[i][y])
            img[i,y] = [255,255,255]
        else:
            segments_fz[i][y] = 0
            img[i,y] = [0,0,0]
# test if the correct segment color was chosen
#print(segmentColor)
# test if the whole hand was extracted
#cv2.imwrite("test2.png",img)

# # calculated contours
# _,cnts,_ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # sort contours from highest to lowest
# cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
# cv2.drawContours(original_image, cnts, 0, (0, 255, 0), 3)
# # make a bounding box to the hand
# # later make the box size constant so that HOGS works well
# x,y,w,h = cv2.boundingRect(cnts[0])
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow("Hand contours with bounding box",original_image)
# close the image window when a key is pressed
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# quit()
ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
plt.show()
