import cv2 as cv
import Detecting
import tkinter.filedialog # To select a file from the file system
import os # To get the current directory while selecting a file

class Selector:
    def __init__(self):
        print("Opening the screen.")
        self.inputPathOfImages = ["",""]
        self.images = ["",""]
        
    def imageSelector(self, imageNumber):
        self.inputPathOfImages[imageNumber-1] = self.filePathFinder()
        self.images[imageNumber-1] = cv.imread(self.inputPathOfImages[imageNumber-1]) # Read the image
        if(self.images[imageNumber-1] is None):
            print("You didn't select a photo")
            cv.destroyAllWindows()
            return

        if( (self.images[0] is not None and self.images[1] is not None)):
            if(len(self.images[0]) != 0 and len(self.images[1]) != 0):
                print("Both images are selected")
                Detecting.Detection(self.images[0], self.images[1]) # Calling correspondence class

    def filePathFinder(self):
        currdir = os.getcwd()
        tempdir = tkinter.filedialog.askopenfilename(initialdir=currdir, title='Please select a file')
        if len(tempdir) > 0:
            print (tempdir)
        return tempdir
