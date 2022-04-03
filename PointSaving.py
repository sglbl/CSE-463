import Homography
import cv2 as cv
import tkinter.filedialog # To select a file from the file system
import os # To get the current directory while selecting a file

class PointSaver:
    def __init__(self, isWebCamMode):
        print("Opening the screen.")
        self.point   = []
        self.counter = 0
        if( isWebCamMode == True ):
            self.inputPath = self.capturer()
        else:
            self.inputPath = self.filePathFinder()
        self.img = cv.imread(self.inputPath) # Read the image
        if(self.img is None):
                print("You didn't select a photo")
                cv.destroyAllWindows()
                return
        cv.namedWindow('image')                         # Naming window in order to use the name for mouse callback function
        cv.setMouseCallback('image', self.savePoints)   # Setting mouse callback function

        while(1):
            cv.imshow('image', self.img)
            k = cv.waitKey(20) & 0xFF                   # Masking the value with 11111111 to get only 8 bits
            if k == 27 or k == ord('q'):                # If the value is 27 (means ESC key is pressed) or q is pressed, then exit.
                break
            if( cv.getWindowProperty('image', cv.WND_PROP_VISIBLE) < 1 ): # If X button is pressed, then exit
                break
        cv.destroyAllWindows()

    def savePoints(self, event, x, y, flag, param):
        cv.putText(self.img, "Click on the northwest corner of the field", (11, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (139,14,252), 1) 
        if event == cv.EVENT_LBUTTONDOWN:
            self.counter = self.counter + 1
            if(self.counter == 1): cv.putText(self.img, "Click on the northeast corner of the field", (11,40), cv.FONT_HERSHEY_PLAIN, 1.0, (139,14,252), 1) 
            if(self.counter == 2): cv.putText(self.img, "Click on the southeast corner of the field", (11,60), cv.FONT_HERSHEY_PLAIN, 1.0, (139,14,252), 1) 
            if(self.counter == 3): cv.putText(self.img, "Click on the southwest corner of the field", (11,80), cv.FONT_HERSHEY_PLAIN, 1.0, (139,14,252), 1) 
            print(f"X coordinate is {x} and y coordinate is {y}")
            self.point.append([x,y])
            cv.circle(self.img, (x,y), 7, (255,0,0), -1)
            print(f"counter = {self.counter}")
            if self.counter == 4:
                print("4 points are selected")
                cv.destroyAllWindows()
                Homography.Homography(self.point, self.inputPath, 'images/modelSoccerField.jpg')
                return
                
    
    def filePathFinder(self):
        currdir = os.getcwd()
        tempdir = tkinter.filedialog.askopenfilename(initialdir=currdir, title='Please select a file')
        if len(tempdir) > 0:
            print (tempdir)
        return tempdir

    def capturer(self):
        internalCameraPortNo = 0
        cam = cv.VideoCapture(internalCameraPortNo)

        while True:
            isTrue, frame = cam.read()

            if isTrue:    
                cv.namedWindow('Video')   # Naming window in order to use the name for X closing button
                
                # Show text on the video
                cv.putText(frame, "Press q to quit or press c to capture image", 
                    (11, 20), # Start point of the text
                    cv.FONT_HERSHEY_PLAIN, # Font
                    1.0, #1.0 for not to scale image
                    (145,200,100), # Color of text
                    1) #1 for the tickness.

                cv.imshow('Video', frame) # Showing the another frame of the video

                if cv.waitKey(20) & 0xFF == ord('e'):
                    break            
                elif( cv.waitKey(20) & 0XFF == ord('c') ):
                    result, image = cam.read()
                    if result:
                        inputPath = 'images/capturing.png'
                        cv.imwrite(inputPath, image)
                        cam.release()   # release the capture pointer  and close the window
                        cv.destroyAllWindows()
                        return inputPath 
                elif( cv.getWindowProperty('Video', cv.WND_PROP_VISIBLE) < 1 ): # If X button is pressed, then exit
                    cam.release()
                    cv.destroyAllWindows()
                    break               
            else:
                cam.release()
                return None