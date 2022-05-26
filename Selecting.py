import cv2 as cv
import Corresponding
import tkinter.filedialog # To select a file from the file system
import os # To get the current directory while selecting a file

class Selector:
    def __init__(self):
        print("Opening the screen.")
        self.inputPathOfImages = ["",""]
        self.images = ["",""]
        self.buttonNo = 1
        self.implementationButtonNo = 2
    
    def buttonSelector(self, buttonNo):
        print("You selected the option " + str(buttonNo.get()))
        self.buttonNo = buttonNo.get()
        
    def implementButtonSelector(self, implementationButtonNo):
        self.implementationButtonNo = implementationButtonNo.get()

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
                Corresponding.Correspondence(self.images[0], self.images[1], self.buttonNo, self.implementationButtonNo) # Calling correspondence class

    def filePathFinder(self):
        currdir = os.getcwd()
        tempdir = tkinter.filedialog.askopenfilename(initialdir=currdir, title='Please select a file')
        if len(tempdir) > 0:
            print (tempdir)
        return tempdir
