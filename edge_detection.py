import time
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

imageloc = sys.argv[1]
img = cv2.imread(imageloc, 0)

laplace = cv2.Laplacian(img, cv2.CV_64F)
sobelx =  cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)

plt.subplot(2, 2, 1),plt.imshow(laplace,cmap = 'gray')
plt.subplot(2, 2, 2),plt.imshow(sobelx,cmap = 'gray')
plt.subplot(2, 2, 3),plt.imshow(sobely,cmap = 'gray')
plt.subplot(2, 2, 4),plt.imshow(sobelxy, cmap = 'gray')

plt.show()