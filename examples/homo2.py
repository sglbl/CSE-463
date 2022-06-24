import cv2 as cv
import numpy as np

class Homography:
    def __init__(self, pointsOfInput, srcImagePath, destImagePath) -> None:
        self.pointsOfInput = pointsOfInput
        self.pointsOfOutput = []

        # Read source image.
        srcImage = cv.imread(srcImagePath)
        print("Points of input")
        print(pointsOfInput)

        # Four corners of the book in source image
        srcPoints = np.array(pointsOfInput, np.float32)

        # Read destination image.
        destImage = cv.imread(destImagePath)

        # Four corners of the book in destination image.
        print(f"Size of destImage(h and w): {destImage.shape} and height: {destImage.shape[0]}, width: {destImage.shape[1]}")
        print("Dsize of destImage is : ", destImage.shape)

        scale = 0.5
        width = int(srcImage.shape[1] * scale)
        height= int(srcImage.shape[0] * scale)
        dimensions = (width,height) #Creating variable called dimensions

        # cv.imshow("VARAN 1SRC" , srcImage)
        # srcImage = cv.resize(srcImage, dimensions, interpolation=cv.INTER_AREA)

        # destPoints = np.array(
        #             [[0, 0],
        #             [srcImage.shape[1]*scale-1, 0],
        #             [srcImage.shape[1]*scale-1, srcImage.shape[0]*scale-1],
        #             [0, srcImage.shape[0]*scale-1]],
        #             np.float32)
        
        #######################

        # destPoints = np.array(
        #             [[0, 0],
        #             [srcImage.shape[1]-1, 0],
        #             [srcImage.shape[1]-1, srcImage.shape[0]-1],
        #             [0, srcImage.shape[0]-1]],
        #             np.float32)

        destPoints = np.array(
                        [[0, 0],
                        [destImage.shape[1]-1, 0],
                        [destImage.shape[1]-1, destImage.shape[0]-1],
                        [0, destImage.shape[0]-1]],
                     np.float32)

        # srcPoints = np.array([[20,20],
        #                      [srcImage.shape[1]-1, 0],
        #                      [srcImage.shape[1]-1, srcImage.shape[0]-1],
        #                      [0, srcImage.shape[0]-1]],
        #                      np.float32)

        # no difference between using `findHomography` and `getPerspectiveTransform` if you're using 4 points as input. 

        # homographyMatrix, status = cv.findHomography(srcPoints, destPoints)
        # resmatrix = cv.getPerspectiveTransform(srcPoints, destPoints)

        # srcPoints = np.array([srcPoints])
        # im_out = cv.perspectiveTransform(srcPoints, resmatrix)
        # print(im_out)

        # Calculate Homography
        
        homographyMatrix, status = cv.findHomography(srcPoints, destPoints)
        print(homographyMatrix)

        # Warp source image to destination based on homography
        im_out = cv.warpPerspective(srcImage, homographyMatrix, (destImage.shape[1], destImage.shape[0])) # width x height

        # Display images
        cv.imshow("Source Image", srcImage)
        cv.imshow("Destination Image", destImage)
        cv.imshow("Warped Source Image", im_out)
        
        cv.waitKey(0)