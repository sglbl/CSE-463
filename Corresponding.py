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
            self.edgeFastBriefNormHammingCorrespondence(image1, image2)

    ####### SIFT WITH BRUTE FORCE ########
    def siftBFCorrespondence(self, image1, image2):
        sift = cv.SIFT_create()

        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

        keyPoints1, descriptors1 = sift.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = sift.detectAndCompute(grayscaleImage2, None)

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
        cv.imshow("My disparity map", disparity_map)

        stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
        dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
        cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)

        plt.imshow(disparity_map, cmap='hot')
        plt.show()

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
        #çç poster to report
        orb = cv.ORB_create()

        keyPoints1, descriptors1 = orb.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = orb.detectAndCompute(grayscaleImage2, None)

        flannMatcher = cv.FlannBasedMatcher_create()

        # converting descriptors to float32 thpe because of flann's knnMatch
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        matches = flannMatcher.knnMatch(descriptors1, descriptors2, k=2)

        # The ratio test to get the good matches
        goodMatches = []
        for m,n in matches:
            if 0.70*n.distance > m.distance:
                goodMatches.append([m]) # adding as array to decode list

        imageOfMatches = cv.drawMatchesKnn(image1, keyPoints1, image2, keyPoints2, goodMatches, None, flags=2)
        cv.imshow("Image of Matches", imageOfMatches)
        
        # Disparity calculation
        height,width = grayscaleImage1.shape
        disparity_map = np.zeros( (height, width) )

        for match in goodMatches:
            (x_img1, y_img1) = keyPoints1[match[0].queryIdx].pt # left image
            (x_img2, y_img2) = keyPoints2[match[0].trainIdx].pt # right image
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
        
        plt.imshow(disparity_map, cmap='hot')
        plt.show()

    ####### STEREO DISPARITY WITH EDGE DETECTOR CANNY / GRADIENT OF GAUSSIAN ########
    def edgeFastBriefNormHammingCorrespondence(self, image1, image2):
        # Creating FastFeatureDetector and BriefDescriptorExtractor
        fast = cv.FastFeatureDetector_create() 
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

        # Turning them into grayscale
        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

        # Edge feature extraction 
        # detecting GRADIENT OF GAUSSIAN (canny)
        edgeImage1 = cv.Canny(grayscaleImage1, 100, 200)
        edgeImage2 = cv.Canny(grayscaleImage2, 100, 200)

        keyPoints1 = fast.detect(edgeImage1, None) #Keypoints of left image
        keyPoints2 = fast.detect(edgeImage2, None) #Keypoints of right image

        keyPoints1, descriptors1 = brief.compute(edgeImage1, keyPoints1)
        keyPoints2, descriptors2 = brief.compute(edgeImage2, keyPoints2)
        
        normHammingMatch = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = normHammingMatch.knnMatch(descriptors1, descriptors2, k=2)
        goodMatches = []
        # Find good matches by using distances
        for m,n in matches:
            if 0.70*n.distance > m.distance:
                goodMatches.append(m)
    
        imageOfMatches= cv.drawMatches(edgeImage1, keyPoints1, edgeImage2, keyPoints2, goodMatches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv.imshow("Image of Matches", imageOfMatches)
        
        # Disparity calculation
        height,width = grayscaleImage1.shape
        disparity_map = np.zeros( (height, width) )

        gtImage = cv.imread("images/poster/disp2.pgm")
        gtImage = cv.cvtColor(gtImage, cv.COLOR_BGR2GRAY)
        # cv.imshow("ground truth", gtImage)


        for match in goodMatches:
            (x_img1, y_img1) = keyPoints1[match.queryIdx].pt # left image
            (x_img2, y_img2) = keyPoints2[match.trainIdx].pt # right image
            x_img1 = int(x_img1);   x_img2 = int(x_img2)
            y_img1 = int(y_img1);   y_img2 = int(y_img2)
            disparity = 8 * abs(x_img2 - x_img1)
            if( disparity > 255 ):
                disparity = 255
            disparity_map[y_img1][x_img1] = disparity
            print("Location on images:", (x_img1, y_img1), (x_img2, y_img2), " Disparity: ", disparity)
            print("Disparity val of ground truth disp " , gtImage[y_img1][x_img1] )

        cv.imshow("Real left image", image1)
        cv.imshow("My disparity map", disparity_map)
        # Ground truth disparity
        stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
        dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
        cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)
        
        plt.imshow(disparity_map, cmap='hot')
        plt.show()