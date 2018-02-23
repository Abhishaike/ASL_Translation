# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
import time
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
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img[::2, ::2]
#print(img)
segments_fz = felzenszwalb(img, scale=1, sigma=0.9, min_size=1000)
#segments_slic = slic(img, n_segments=10, compactness=10, sigma=1)
# segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# gradient = sobel(rgb2gray(img))
# segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
# print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
# print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,
                 subplot_kw={'adjustable': 'box-forced'})
# returns the index of unique segments in a flattened array
_,index_Uniques = np.unique(segments_fz,return_index = True)
# flattens the original image array so that it can be compared with the uniques
flat_image = np.ravel(img)
# segment color is the color of the hand
segmentColor = None
has_color = False
for i in range(len(img)):
    for j in range(len(img[i])):
        # chooses the segment that has the color of the hand
        #print(img[i,j])
        if(abs(img[i,j][0] - 148 < 12)and abs(img[i,j][1] - 95 < 25) and abs(img[i,j][2]- 200 < 25)):
            segmentColor = segments_fz[i,j]
            has_color = True
    if(has_color):
        break
for i in range(len(segments_fz)):
    for y in range(len(segments_fz[i])):
        if(segments_fz[i][y]==7):
            segments_fz[i][y] = 1
            print(img[i][y])
            img[i,y] = [255,255,255]
        else:
            segments_fz[i][y] = 0
            img[i,y] = [0,0,0]

print(segmentColor)
#print(mark_boundaries(img, segments_fz))
ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")

# k means
# image = cv2.GaussianBlur(image, (11, 11), 2)
# Z=image.reshape((-1,3))
# Z=np.float32(Z)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .5)
# K=int(args["kvalue"])
# attempts=1
# start = time.time()
# ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
# end = time.time()
# print(end-start)
# center = np.uint8(center)
# print(center)
# res=center[label.flatten()]
# res2=res.reshape((image.shape))
#
# #uncomment if you want to binarize
# start = time.time()
# purple_matrix = np.full(len(res2),len(res2[0]),[36,71,214])
# for i in range(len(res2)):
#     i-=1
#     for y in range(len(res2[i])):
#         y-=1
# #        print(y)
#         if(np.array_equal(res2[i][y],np.array([36,71,214]))): #MUST CHANGE THIS TO PURPLE COLOR
#         #if(abs(res2[i][y][0] - 10) < 25 and abs(res2[i][y][1] - 23) < 25 and abs(res2[i][y][2] - 241)<25):
#             res2[i][y] = np.array([255,255,255])
#         else:
#             res2[i][y]=np.array([0,0,0])
# end = time.time()
# print(end-start)
# gray = res2
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
#
#
# #don't fuck with this unless you want to do some contour shit
# #threshold the image, then perform a series of erosions +
# #dilations to remove any small regions of noise
# thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=2)
#
# # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# # c = max(cnts, key=cv2.contourArea)
# #
# # extLeft = tuple(c[c[:, :, 0].argmin()][0])
# # extRight = tuple(c[c[:, :, 0].argmax()][0])
# # extTop = tuple(c[c[:, :, 1].argmin()][0])
# # extBot = tuple(c[c[:, :, 1].argmax()][0])
#
# # denoise the image
# initial_kernel = np.ones((12,12),np.uint8)
# closer_kernel = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, initial_kernel)
# #cv2.imwrite('binary_output_withMorphology.png',gray)
# cv2.imshow('res2',gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows();
# quit();
