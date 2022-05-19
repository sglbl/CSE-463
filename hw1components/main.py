import cv2 as cv
import numpy as np
from PointSaver import *
# from CamCapture import *

def checkIfButtonClicked(event, x, y, flag, param):
    if (event == cv.EVENT_LBUTTONDOWN): # If left button is clicked
        if( x > 99 and x < 501 ):
            if( y > 124 and y < 176 ):
                print("Cam")
                # return image path of cam if success
            elif( y > 224 and y < 276 ):
                print("Folder")
                # Folder selection
                inputPath = 'images/input1.png'
                # Create a window to select 4 points.
                PointSaver(inputPath) # Calling point saver

if __name__=='__main__':
    # creating a blank image
    # uint8 is datatype of an image.
    # shape of 3 is for the number of color channels.
    blank = np.zeros((300,600,3), dtype ='uint8')     

    blank[0:100] = (0,100,250)     # Top row
    blank[100:300] = (205,206,207) # Paint the background color to gray

    cv.putText(blank, "Suleyman's Homograpy App", 
                (50, 50), # Starting point of the text
                cv.FONT_HERSHEY_TRIPLEX,
                1.0, #1.0 for not to scale image
                (0,0,0), # Color of the text
                2) #2 for the tickness.

    cv.rectangle(blank, (100,125), (500,175), (0,0,255), thickness = 2)
    cv.putText(blank, "Click to get image from Webcam", 
                (150, 150), # Start point of the text
                cv.FONT_HERSHEY_PLAIN, # Font
                1.0, #1.0 for not to scale image
                (0,0,0), # Color of text
                1) #1 for the tickness.

    cv.rectangle(blank, (100,225), (500,275), (0,0,255), thickness = 2)
    cv.putText(blank, "Click to get image from Folder", 
                (150, 250), # Start point of the text
                cv.FONT_HERSHEY_PLAIN, # Font
                1.0, #1.0 for not to scale image
                (0,0,0), # Color of text
                1) #1 for the tickness.

    cv.namedWindow('Main Page')                         # Naming window in order to use the name for mouse callback function
    cv.setMouseCallback('Main Page', checkIfButtonClicked)   # Setting mouse callback function
    cv.imshow('Main Page', blank)

    cv.waitKey(0)