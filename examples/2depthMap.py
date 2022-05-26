import cv2 as cv
from matplotlib import pyplot as plt

DEPTH_VISUALIZATION_SCALE = 2048

imgL = cv.imread('flowers-left.png',0)
imgR = cv.imread('flowers-right.png',0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# NUMDISPARITIES	the disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) 
#                   to numDisparities. The search range can then be shifted by changing the minimum disparity.
# BLOCKSIZE	the linear size of the blocks compared by the algorithm. 
#           The size should be odd (as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. 
#           Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.


disparity = stereo.compute(imgL,imgR) #depth
cv.imshow('depth/255', disparity / 255)
cv.imshow('depth', disparity )
# print(disparity[50])

cv.waitKey(0)
# plt.imshow(disparity,'gray')
# plt.show()