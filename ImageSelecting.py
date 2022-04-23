import cv2 as cv
import ObjectDetecting
import tkinter.filedialog # To select a file from the file system
import os # To get the current directory while selecting a file

class ImageSelector:
    def __init__(self):
        print("Opening the screen.")
        self.inputPathOfImages = ["",""]
        self.images = ["",""]
        
    def selector(self, imageNumber):
        self.inputPathOfImages[imageNumber-1] = self.filePathFinder()
        self.images[imageNumber-1] = cv.imread(self.inputPathOfImages[imageNumber-1]) # Read the image
        if(self.images[imageNumber-1] is None):
            print("You didn't select a photo")
            cv.destroyAllWindows()
            return

        if( (self.images[0] is not None and self.images[1] is not None)):
            if(len(self.images[0]) != 0 and len(self.images[1]) != 0):
                print("Both images are selected")
                ObjectDetecting.ObjectDetector(self.images[0], self.images[1]) # Calling object detector

            # cv.namedWindow('image')  # Putting name to window

            # Showing the image
            # while(1):
            #     cv.imshow('image', self.images[imageNumber-1])
            #     k = cv.waitKey(20) & 0xFF                   # Masking the value with 11111111 to get only 8 bits
            #     if k == 27 or k == ord('q'):                # If the value is 27 (means ESC key is pressed) or q is pressed, then exit.
            #         break
            #     if( cv.getWindowProperty('image', cv.WND_PROP_VISIBLE) < 1 ): # If X button is pressed, then exit
            #         break
            # cv.destroyAllWindows()

    def filePathFinder(self):
        currdir = os.getcwd()
        tempdir = tkinter.filedialog.askopenfilename(initialdir=currdir, title='Please select a file')
        if len(tempdir) > 0:
            print (tempdir)
        return tempdir