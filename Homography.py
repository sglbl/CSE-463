import cv2 as cv
import numpy as np       # Mathematical array operations
import scipy.interpolate # to interpolate a matrix
import tkinter as tk

class Homography:
    def __init__(self, pointsOfInput, srcImagePath, destImagePath) -> None:
        self.pointsOfInput = pointsOfInput
        self.pointsOfOutput = []

        # Read source image.
        srcImage = cv.imread(srcImagePath)
        # Four corners of the book in source image
        srcPoints = np.array(pointsOfInput, np.float32)

        # Read destination image.
        destImage = cv.imread(destImagePath)

        # Four corners of the book in destination image.
        print(f"Size of destination image (h and w): {destImage.shape}")

        destPoint1 = [1, 1];     destPoint2 = [destImage.shape[1]-1, 1];     destPoint3 = [destImage.shape[1]-1, destImage.shape[0]-1];
        destPoints = np.array(
                        [   destPoint1,
                            destPoint2,
                            destPoint3, # [1, 700/383]
                            self.find4thPointByIntersectionOfParallelLines(destPoint1, destPoint2, destPoint3)
                        ], np.float32)
        
        homographyMatrix, status = cv.findHomography(srcPoints, destPoints) # Homography matrix

        final = self.sg_warpPerspective(srcImage, homographyMatrix, int(destImage.shape[1]), int(destImage.shape[0])) # width x height

        # Display images
        # cv.imshow("Destination Image", destImage)
        self.showImagesTogether(srcImage, final)

        cv.waitKey(0)

    def showImagesTogether(self, srcImage, final):
        # First, making images same height
        if( final.shape[0] >= srcImage.shape[0] ): # If the height of the final image is greater than the height of the src image
            srcImage = cv.copyMakeBorder(srcImage, 0, final.shape[0]-srcImage.shape[0], 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])    
        else:
            final = cv.copyMakeBorder(final, 0, srcImage.shape[0]-final.shape[0], 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
        concatHorizontally = np.concatenate((srcImage, final), axis=1) # x axis
    
        texts = np.zeros((45, concatHorizontally.shape[1],3), dtype ='uint8')     # Making a blank image of same width as concatHorizontally

        # 5. Write text on image
        cv.putText(texts, 'Input Image', (25,25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (140,40,120), 1) 
        cv.putText(texts, 'Output Model Field', (srcImage.shape[1] + 20 , 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (140,40,120), 1) 

        concatVertically = np.concatenate((concatHorizontally, texts), axis=0) # y axis
        cv.imshow("Vertical Concatenation", concatVertically)

    def find4thPointByIntersectionOfParallelLines(self, destPoint1, destPoint2, destPoint3):
        # Finding the 4th point with projective intersection of the two parallel lines
        '''
        P1      P2
         ________
        |        |
        |        |              1) using P1 value from P1P2 line to get the first coordinate vector of the line
        |        |              2) using P3 value from P3P4 line to get the second coordinate vector of the line
        |        |              P1 and P4 are in the lines that are parallel to each other
        P4______P3              3) using the intersection of the two lines to get the 4th point [using determinant formula]
        '''
        # ax + by + c = 0
        a1,b1,c1 = self.findLineEquationFromPoint('b',destPoint1)  # Assume b is zero for point1
        a2,b2,c2 = self.findLineEquationFromPoint('a',destPoint3)  # Assume a is zero for point3
        coefficientOfI, coefficientOfJ = self.calculateDeterminant(a1,b1,c1, a2,b2,c2)
        return [coefficientOfI, coefficientOfJ]

    def calculateDeterminant(self, a1,b1,c1, a2,b2,c2):
        # | i  j  k  |
        # | a1 b1 c1 |
        # | a2 b2 c2 |
        coefficientOfI = b1*c2 - b2*c1  # x value
        coefficientOfJ = a2*c1 - a1*c2  # y value
        return coefficientOfI, coefficientOfJ

    def findLineEquationFromPoint(self, pick0, point):
        # ax + by + c = 0
        if(pick0 == 'b'):  # If a point of vertical line, assume b = 0
            # ax = -c
            # c = -ax
            a = 1 # To find rate of a and c assume a = 1
            b = 0
            c = -1*a*point[0]  # point[0] is x value
            return a,b,c
        elif(pick0 == 'a'):
            # by = -c
            # c = -by
            b = 1
            a = 0
            c = -1*b*point[1]  # point[1] is y value
            return a,b,c

    def interpolateTheImage(self, matrix):
        image = np.zeros((matrix.shape[1], matrix.shape[0], matrix.shape[2]), dtype='uint8') # width, height and color channels

        print("Interpolating")
        for i in range(matrix.shape[0]): # for every row of height
            # matrix[i] is a row of the matrix which is 2d.
            y,x = np.where(matrix[i]!=0)   # If matrix[i] is not zero, then get the index of non-zero values
            xVector = np.linspace(np.min(x), np.max(x), 3) # create evenly spaced sample number of dimension(3) times.
            yVector = np.linspace(np.min(y), np.max(y), matrix.shape[1]) # Create evenly spaced sample 'width'(matrix.shape[1]) number of points
            xCoordMatrix, yCoordMatrix = np.meshgrid(xVector, yVector)   # Return coordinate matrices from coordinate vectors xx and yy
            matrix[i] = scipy.interpolate.griddata( (x,y), matrix[i][matrix[i]!=0], (xCoordMatrix, yCoordMatrix), method='nearest') 
            image[:,i] = matrix[i]   # 2d matrix[i] is a row of the image matrix

        return image

    def sg_warpPerspective(self, srcImage, homographyMatrix, widthOfWindow, heightOfWindow):
        matrix = np.zeros((widthOfWindow, heightOfWindow, srcImage.shape[2]))
        for i in range(srcImage.shape[1]):  # width
            for j in range(srcImage.shape[0]): # height
                coordinateVector = np.dot(homographyMatrix, [i,j,1])
                iOfNewMatrix, jOfNewMatrix,_ = (coordinateVector / coordinateVector[2] + 0.4 ) # 0.4 is to make the image look more smooth
                iOfNewMatrix, jOfNewMatrix = int(iOfNewMatrix), int(jOfNewMatrix) # Converting coordinate vector elements to integer because they will be used as indexes.
                if iOfNewMatrix >= 0 and iOfNewMatrix < widthOfWindow:  # If the index is in the range of the new matrix
                    if jOfNewMatrix >= 0 and jOfNewMatrix < heightOfWindow: # If the index is in the range of the new matrix
                        matrix[iOfNewMatrix, jOfNewMatrix] = srcImage[j,i]
        
        return self.interpolateTheImage(matrix)