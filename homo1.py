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
        destPoints = np.array(
                        [[0, 0],
                        [destImage.shape[1]-1, 0],
                        [destImage.shape[1]-1, destImage.shape[0]-1],
                        [0, destImage.shape[0]-1]],
                     np.float32)

        # no difference between using `findHomography` and `getPerspectiveTransform` if you're using 4 points as input. 

        # homographyMatrix, status = cv.findHomography(srcPoints, destPoints)
        # resmatrix = cv.getPerspectiveTransform(srcPoints, destPoints)

        # srcPoints = np.array([srcPoints])
        # im_out = cv.perspectiveTransform(srcPoints, resmatrix)
        # print(im_out)

        # Calculate Homography
#        srcPoints = np.array([[20,20], [300,20], [300,300], [20,300]], np.float32)
        homographyMatrix, status = cv.findHomography(srcPoints, destPoints)
        print(homographyMatrix)

        # Warp source image to destination based on homography
        im_out = cv.warpPerspective(srcImage, homographyMatrix, (destImage.shape[1], destImage.shape[0])) # width x height

        # Display images
        cv.imshow("Source Image", srcImage)
        cv.imshow("Destination Image", destImage)
        cv.imshow("Warped Source Image", im_out)
        
        cv.waitKey(0)