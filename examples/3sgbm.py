import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

leftImage = cv.imread('flowers-left.png')
rightImage = cv.imread('flowers-right.png')

leftImageGray = cv.cvtColor(leftImage, cv.COLOR_BGR2GRAY)
rightImageGray = cv.cvtColor(rightImage, cv.COLOR_BGR2GRAY)

stereo_bm = cv.StereoBM_create(32)
dispmap_bm = stereo_bm.compute(leftImageGray, rightImageGray)

stereo_sgbm = cv.StereoSGBM_create(0, 32)
dispmap_sgbm = stereo_sgbm.compute(leftImage, rightImage)

# plt.figure(figsize=(12,10))
# plt.subplot(221)
# plt.title('left')

plt.imshow(dispmap_bm,'gray')
plt.show()
plt.imshow(dispmap_sgbm, 'gray')
plt.show()
