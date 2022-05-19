
# importing OpenCV library
 # from cv2 import *
import cv2 as cv
  
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that
internalCameraPortNo = 0
cam = cv.VideoCapture(internalCameraPortNo)

while True:
    isTrue, frame = cam.read()

    if isTrue:    
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
        elif( cv.waitKey(20) & 0XFF == ord('s') ):
            result, image = cam.read()
            if result:
                cv.imwrite('images/capturing.png', image)
                break
    else:
        break

cam.release()   #release the capture pointer  


# # reading the input using the camera
# result, image = cam.read()

# # If image will detected without any error, 
# # show result
# if result:
#     cv.imshow("GeeksForGeeks", image)
#     cv.imwrite("images/inputFromWebCam.png", image) # Store photo
  
#     # If keyboard interrupt occurs, destroy image 
#     # window
#     cv.waitKey(0)
#     cv.destroyWindow("GeeksForGeeks")
  
# # If captured image is corrupted, moving to else part
# else:
#     print("No image detected. Please! try again")