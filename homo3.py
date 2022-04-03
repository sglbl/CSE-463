import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# To open matplotlib in interactive mode
%matplotlib qt
 
# Load the image
img = cv2.imread('images/input1.png') 
 
# Create a copy of the image
img_copy = np.copy(img)

# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the 
# transformation matrix
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
 
plt.imshow(img_copy)
 
# Specify input and output coordinates that is used
# to calculate the transformation matrix
input_pts = np.float32([[80,1286],[3890,1253],[3890,122],[450,115]])
output_pts = np.float32([[100,100],[100,3900],[2200,3900],[2200,100]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)

# Apply the perspective transformation to the image
out = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

# Display the transformed image
plt.imshow(out)

cv2.waitKey(0)
