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
            self.cornerCorrespondence(image1, image2)    

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

    ####### STEREO DISPARITY WITH HARRIS CORNER ########
    def cornerCorrespondence(self, image1, image2):
        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)
        coordinates1 = self.harris(grayscaleImage1)
        coordinates2 = self.harris(grayscaleImage2)

        # convert coordinates to Keypoint type
        keyPoints1 = [cv.KeyPoint(crd[0], crd[1], 13) for crd in coordinates1]
        # print("Keypoints are", keyPoints1)

        # compute SIFT descriptors from corner keypoints
        sift = cv.SIFT_create()
        # print("first desc is", sift.compute(grayscaleImage1, coordinates1[0])[1]  )
        descriptors1 = [sift.compute(grayscaleImage1, [kp])[1] for kp in coordinates1]
        # descriptors1 = sift.compute(image1, keyPoints1)
        print("type of desc is ", type(descriptors1), "\ndesc is \n", descriptors1)

        ##########################################################

        keyPoints2 = [cv.KeyPoint(crd[0], crd[1], 13) for crd in coordinates2]

        # compute SIFT descriptors from corner keypoints
        sift = cv.SIFT_create()
        descriptors2 = [sift.compute(grayscaleImage2, [kp])[1] for kp in coordinates2]
        # descriptors2 = sift.compute(image2, keyPoints2)

        goodMatches = self.matcherAndSorter(descriptors1, descriptors2)
        # matches = self.goodMatchFinder(descriptors1, descriptors2)
        # goodMatches = matches
        
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


        


    def harris(self,imagePath):       
        # Read in the image
        image = imagePath #çç

        # Make a copy of the image
        image_copy = np.copy(image)

        # Change color to RGB (from BGR)
        image_copy = cv.cvtColor(image_copy, cv.COLOR_BGR2RGB)

        plt.imshow(image_copy)

        # ### Detect corners

        # Convert to grayscale
        gray = cv.cvtColor(image_copy, cv.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        # Detect corners 
        dst = cv.cornerHarris(gray, 2, 3, 0.04)

        # Dilate corner image to enhance corner points
        dst = cv.dilate(dst,None)

        cv.imshow("dst", dst)
        # plt.imshow(dst, cmap='gray')
        # plt.show()

        # ### Extract and display strong corners

        # This value vary depending on the image and how many corners you want to detect
        # Try changing this free parameter, 0.1, to be larger or smaller and see what happens
        thresh = 0.1*dst.max()

        # Create an image copy to draw corners on
        corner_image = np.copy(image_copy)

        ijValues = []
        # Iterate through all the corners and draw them on the image (if they pass the threshold)
        for j in range(0, dst.shape[0]):
            for i in range(0, dst.shape[1]):
                if(dst[j,i] > thresh):
                    # image, center pt, radius, color, thickness
                    ijValues.append( (i,j) ) 
                    cv.circle( corner_image, (i, j), 1, (0,255,0), 1)

        # plt.imshow(corner_image)
        # plt.show()
        cv.imshow("Corner img", corner_image)
        return ijValues
