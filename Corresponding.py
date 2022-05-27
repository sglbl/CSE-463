import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Correspondence:
    def __init__(self, image1, image2, featureButtonNo) -> None:
        if(featureButtonNo == 1):
            self.siftBFCorrespondence(image1, image2)
        if(featureButtonNo == 2):
            self.orbFlannCorrespondence(image1, image2)
        if(featureButtonNo == 3):
            self.edgeCorrespondence(image1, image2)    

    ####### SIFT WITH BRUTE FORCE ########
    def siftBFCorrespondence(self, image1, image2):
        sift = cv.SIFT_create()

        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

        keyPoints1, descriptors1 = sift.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = sift.detectAndCompute(grayscaleImage2, None)

        # matches = self.matcherAndSorter(descriptors1, descriptors2)
        matches = self.goodMatchFinder(descriptors1, descriptors2)
        imageOfMatches= cv.drawMatches(grayscaleImage1, keyPoints1, grayscaleImage2, keyPoints2, matches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow("Image of Matches", imageOfMatches)
    
        height,width = grayscaleImage1.shape
        disparity_map = np.zeros( (height, width) )

        for match in matches:
            (x_img1, y_img1) = keyPoints1[match.queryIdx].pt # left image
            (x_img2, y_img2) = keyPoints2[match.trainIdx].pt # right image
            x_img1 = int(x_img1);   x_img2 = int(x_img2)
            y_img1 = int(y_img1);   y_img2 = int(y_img2)
            disparity = 8*abs(x_img2 - x_img1)
            if( disparity > 255 ):
                disparity = 255
            disparity_map[y_img1][x_img1] = disparity
            # cv.circle(disparity_map, (y_img1, x_img1), 1, # draw a circle with raidus 1 on (y_img1, x_img1) location
            #         disparity, thickness=cv.FILLED) # fill the circle, and make the color of it abs(x_img2 - x_img1) on graysale

            print("Location on images:", (x_img1, y_img1), (x_img2, y_img2), " Disparity: ", disparity)

        cv.imshow("Real left image", image1)
        # plt.imshow(disparity_map, cmap='hot')
        # plt.show()

        cv.imshow("My disparity map", disparity_map)

        stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
        dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
        cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)

    def goodMatchFinder(self, descriptor1, descriptor2):
        # Using brute for match
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

    def matcherAndSorter(self, descriptor1, descriptor2):
        matches = cv.BFMatcher().match(descriptor1, descriptor2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return matches
    
    ####### ORB WITH FLANN FORCE ########
    def orbFlannCorrespondence(self, image1, image2):
        matcher = cv.FlannBasedMatcher_create()
        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)
        
        orb = cv.ORB_create()

        keyPoints1, descriptors1 = orb.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = orb.detectAndCompute(grayscaleImage2, None)

        # Creating matcher
        # matches = self.matcherAndSorter(descriptors1, descriptors2)
        matches = self.goodMatchFinder(descriptors1, descriptors2)
        imageOfMatches= cv.drawMatches(grayscaleImage1, keyPoints1, grayscaleImage2, keyPoints2, matches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("Image of Matches", imageOfMatches)
        
        # Disparity calculation
        height,width = grayscaleImage1.shape
        disparity_map = np.zeros( (height, width) )

        for match in matches:
            (x_img1, y_img1) = keyPoints1[match.queryIdx].pt # left image
            (x_img2, y_img2) = keyPoints2[match.trainIdx].pt # right image
            x_img1 = int(x_img1);   x_img2 = int(x_img2)
            y_img1 = int(y_img1);   y_img2 = int(y_img2)
            disparity = 8*abs(x_img2 - x_img1)
            if( disparity > 255 ):
                disparity = 255
            disparity_map[y_img1][x_img1] = disparity
            print("Location on images:", (x_img1, y_img1), (x_img2, y_img2), " Disparity: ", disparity)

        cv.imshow("Real left image", image1)
        cv.imshow("My disparity map", disparity_map)
        # Ground truth disparity
        stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
        dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
        cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)

