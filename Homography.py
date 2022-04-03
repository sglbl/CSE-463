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

        scale = 0.80
        width = int(srcImage.shape[1] * scale)
        height= int(srcImage.shape[0] * scale)
        dimensions = (width,height) #Creating variable called dimensions

        destPoints = np.array(
                        [[0, 0],
                        [destImage.shape[1]-1, 0],
                        [destImage.shape[1]-1, destImage.shape[0]-1],
                        [0, destImage.shape[0]-1]],
                     np.float32)
        
        homographyMatrix, status = cv.findHomography(srcPoints, destPoints)

        ####
        # centerOfSrcImage = [int(srcImage.shape[1]/2), int(srcImage.shape[0]/2)] + [1]
        # centerOfSrcImageAfterWrap = np.dot(homographyMatrix, centerOfSrcImage )
        # centerOfSrcImageAfterWrap = [x/centerOfSrcImageAfterWrap[2] for x in centerOfSrcImageAfterWrap]
        
        # centerOfWrappedOutputImage = srcImage.shape
        # x_offset = centerOfWrappedOutputImage[0] - centerOfSrcImageAfterWrap[0]
        # y_offset = centerOfWrappedOutputImage[1] - centerOfSrcImageAfterWrap[1]

        # # Tranlastion matrix
        # translationMatrix = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]], np.float32)
        # # translated homography matrix
        # translatedHomographyMatrix = np.dot(translationMatrix, homographyMatrix)
        # # warp
        # stitched = cv.warpPerspective(srcImage, translatedHomographyMatrix, (destImage.shape[1], destImage.shape[0]))
        # cv.imshow("Stitch", stitched)
        ####

        im_out = cv.warpPerspective(srcImage, homographyMatrix, (int(destImage.shape[1]), int(destImage.shape[0]))) # width x height
        cv.perspectiveTransform(srcPoints, destPoints, homographyMatrix)

        # Display images
        cv.imshow("Source Image", srcImage)
        cv.imshow("Destination Image", destImage)
        cv.imshow("Warped Source Image", im_out)
        
        cv.waitKey(0)