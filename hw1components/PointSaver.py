import cv2 as cv
  
class PointSaver:
    def __init__(self, inputPath):
        print("Opening the screen.")
        self.point   = []
        self.counter = 0
        self.img = cv.imread(inputPath) 
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
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img, (x,y), 7, (255,0,0), -1)
            print(f"X coordinate is {x} and y coordinate is {y}")
            self.counter = self.counter + 1
            self.point.append([x,y])
            print(f"counter = {self.counter}")
            if self.counter == 4:
                print("4 points are selected")
