import cv2 as cv
import numpy as np

class Detection:
    def __init__(self, image1, image2, buttonNo) -> None:
        if(buttonNo == 1):
            self.siftBFCorrespondence(image1, image2)
        elif(buttonNo == 2):
            self.orbFlannCorrespondence(image1, image2)

    ####### SIFT WITH BRUTE FORCE ########
    def siftBFCorrespondence(self, image1, image2):
        sift = cv.SIFT_create()

        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

        keyPoints1, descriptors1 = sift.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = sift.detectAndCompute(grayscaleImage2, None)

        goodMatches = self.goodMatchFinder(descriptors1, descriptors2)
        imageOfMatches= cv.drawMatches(image1, keyPoints1, image2, keyPoints2, goodMatches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # show goodmatches
        cv.imshow("Matches", imageOfMatches)

        if len(goodMatches) > 7:
            srcPoints = []  # query values
            destPoints = [] # destination values
            for match in goodMatches:
                (x_img1, y_img1) = keyPoints1[match.queryIdx].pt
                (x_img2, y_img2) = keyPoints2[match.trainIdx].pt
                srcPoints.append(  [(x_img1, y_img1)] )
                destPoints.append( [(x_img2, y_img2)] )
            srcPoints = np.float32(srcPoints);  # Converting to numpy array
            destPoints = np.float32(destPoints);  # Converting to numpy array
            srcPoints.reshape(-1, 1, 2)
            destPoints.reshape(-1, 1, 2)

            matrix, _ = cv.findHomography(srcPoints, destPoints, cv.RANSAC, 4.9)
            # print("Srcpts[0] is", srcPoints[0])
            # Perspective transformation with homography
            height, width , _ = image1.shape
            # Using max window size to find points
            newPoints = self.destFinder(height, width, matrix)
            
            # transparent overlays
            newImg=image2.copy()
            homography = cv.fillConvexPoly(image2, newPoints, (214, 183, 37))
            cv.addWeighted(homography, 0.4, newImg, 0.6, 0, newImg)

            cv.imshow("Homography", newImg)
        else:
            cv.imshow("HomographyNo", image2)

    def destFinder(self, height, width, matrix):
        pts = np.float32(   [[0, 0],
                            [0, height], 
                            [width, height], 
                            [width, 0]]
                            ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, matrix)
        dst = np.array(dst, np.int32)
        return dst

    def goodMatchFinder(self, descriptor1, descriptor2):
        matches = cv.BFMatcher().knnMatch(descriptor1, descriptor2, k=2)
        goodMatches = []
        # Find good matches by using distances
        for m,n in matches:
            if 0.70*n.distance > m.distance:
                goodMatches.append(m)
        return goodMatches    

    def imagesToGrayscale(self, image1, image2):
        # Converting images to grayscale for detection
        image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        return image1, image2

    ####### ORB WITH FLANN FORCE ########
    def orbFlannCorrespondence(self, image1, image2):
        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)
        orb = cv.ORB_create()

        keyPoints1, descriptors1 = orb.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = orb.detectAndCompute(grayscaleImage2, None)

        goodMatches = self.goodMatchFinder(descriptors1, descriptors2)
        # imageOfMatches = cv.drawMatches(grayscaleImage1, keyPoints1, grayscaleImage2, keyPoints2, goodMatches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if len(goodMatches) > 7:
            srcPoints = []  # query values
            destPoints = [] # destination values
            for match in goodMatches:
                (x_img1, y_img1) = keyPoints1[match.queryIdx].pt
                (x_img2, y_img2) = keyPoints2[match.trainIdx].pt
                srcPoints.append(  [(x_img1, y_img1)] )
                destPoints.append( [(x_img2, y_img2)] )
            srcPoints = np.float32(srcPoints);  # Converting to numpy array
            destPoints = np.float32(destPoints);  # Converting to numpy array
            srcPoints.reshape(-1, 1, 2)
            destPoints.reshape(-1, 1, 2)
            matrix, _ = cv.findHomography(srcPoints, destPoints, cv.RANSAC, 4.9)
            # print("Srcpts[0] is", srcPoints[0])
            # Perspective transformation with homography
            height, width , _ = image1.shape
            # Using max window size to find points
            newPoints = self.destFinder(height, width, matrix)
            
            # transparent overlays
            newImg=image2.copy()
            homography = cv.fillConvexPoly(image2, newPoints, (214, 183, 37))
            cv.addWeighted(homography, 0.4, newImg, 0.6, 0, newImg)

            cv.imshow("Homography", newImg)
        else:
            cv.imshow("HomographyNo", image2)
