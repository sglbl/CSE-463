import cv2 as cv

def capturer():
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

    cam.release()   # release the capture pointer  and close the window