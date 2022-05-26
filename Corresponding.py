import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Correspondence:
    def __init__(self, image1, image2, featureButtonNo, implementationButtonNo) -> None:
        if(featureButtonNo == 1):
            self.siftBFCorrespondence(image1, image2, implementationButtonNo)
        if(featureButtonNo == 2):
            self.orbFlannCorrespondence(image1, image2, implementationButtonNo)
        if(featureButtonNo == 3):
            self.edgeCorrespondence(image1, image2, implementationButtonNo)    

    def siftBFCorrespondence(self, image1, image2, implementationButtonNo):
        sift = cv.SIFT_create()

        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

        keyPoints1, descriptors1 = sift.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = sift.detectAndCompute(grayscaleImage2, None)

        matches = self.matcherAndSorter(descriptors1, descriptors2)
        imageOfMatches= cv.drawMatches(grayscaleImage1, keyPoints1, grayscaleImage2, keyPoints2, matches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow("Image of Matches", imageOfMatches)
        
        if(implementationButtonNo == 1): # OpenCV function
            # Stereo correspondence with block matching algorithm
            stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
            dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
            cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)
            cv.waitKey(0)
        if(implementationButtonNo == 2): # My implementation
            i = 0
            blockSize = 5    
            height,width = grayscaleImage1.shape
            disparity_map = np.zeros(grayscaleImage1.shape)

            for match in matches:
                (x_img1, y_img1) = keyPoints1[match.queryIdx].pt
                (x_img2, y_img2) = keyPoints2[match.trainIdx].pt
                i += 1
                # disparity_map[(int)(y_img1) - blockSize: (int)(y_img1) + blockSize][(int)(x_img1) - blockSize : (int)(x_img1) + blockSize] = (x_img2 - x_img1)
                disparity_map[(int)(y_img1)][(int)(x_img1)] = (int)(abs(x_img2 - x_img1))
                print(i, (x_img1, y_img1), (x_img2, y_img2), abs(x_img2-x_img1))

            cv.imshow("Real image", image1)

            # plt.imshow(disparity_map, cmap='jet')
            # plt.show()

            cv.imshow("disp map", disparity_map)

            stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
            dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
            cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)
            # baseline = 12.5   # cam distance is 12.5 pixels
            # fx = 942.8  
            # depth = fx * baseline / disparity




    

    # def siftBFCorrespondence2(self, image1, image2, implementationButtonNo):
    #     sift = cv.SIFT_create()

    #     grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

    #     keyPoints1, descriptors1 = sift.detectAndCompute(grayscaleImage1, None)
    #     keyPoints2, descriptors2 = sift.detectAndCompute(grayscaleImage2, None)

    #     matches = self.matcherAndSorter(descriptors1, descriptors2)
    #     imageOfMatches= cv.drawMatches(grayscaleImage1, keyPoints1, grayscaleImage2, keyPoints2, matches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #     cv.imshow("Image of Matches", imageOfMatches)
        
    #     if(implementationButtonNo == 1): # OpenCV function
    #         # Stereo correspondence with block matching algorithm
    #         stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
    #         dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
    #         cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)
    #         cv.waitKey(0)
    #     if(implementationButtonNo == 2): # My implementation

    #         grayscaleImage1 = cv.resize(grayscaleImage1, (int(grayscaleImage1.shape[1] / 2), int(grayscaleImage1.shape[0] / 2)))
    #         grayscaleImage2 = cv.resize(grayscaleImage2, (int(grayscaleImage2.shape[1] / 2), int(grayscaleImage2.shape[0] / 2)))
    #         image1 = grayscaleImage1
    #         image2 = grayscaleImage2

    #         # pointsInImage1 = []
    #         # pointsInImage2 = []
    #         # i = 0;
    #         # blockSize = 15        
    #         # # height,width,_ = image1.shape
    #         # # disparity_map = np.ones((height, width, 3), dtype ='uint8')     
    #         # for match in matches:
    #         #     (x_img1, y_img1) = keyPoints1[match.queryIdx].pt
    #         #     (x_img2, y_img2) = keyPoints2[match.trainIdx].pt
    #         #     if(1==1):
    #         #     # if( (int)(y_img1) == (int)(y_img2) ):
    #         #         pointsInImage1.append((x_img1, y_img1))
    #         #         pointsInImage2.append((x_img2, y_img2))
    #         #         i += 1
    #         #         # disparity_map[((int)(y_img1) - blockSize): ((int)(y_img1) + blockSize)][(int)(x_img1)] = (x_img2 - x_img1)
    #         #         disparity_map[(int)(y_img1)][(int)(x_img1)] = (x_img2 - x_img1)
    #         #         print(i, (x_img1, y_img1), (x_img2, y_img2))
                
    #         sift = cv.SIFT_create()
    #         keypoints = []
    #         for i in range(image1.shape[0]):
    #             for j in range(image1.shape[1]):
    #                 keypoints.append(cv.KeyPoint(x=j, y=i, size=12))
    #         keypoints_returned, descriptorsNew1 = sift.compute(image1, keypoints)
    #         descriptorsNew1 = np.asarray(descriptorsNew1).reshape((image1.shape[0], image1.shape[1], 128))
            
    #         sift = cv.SIFT_create()
    #         keypoints = []
    #         for i in range(image2.shape[0]):
    #             for j in range(image2.shape[1]):
    #                 keypoints.append(cv.KeyPoint(x=j, y=i, size=12))
    #         keypoints_returned, descriptorsNew2 = sift.compute(image2, keypoints)
    #         descriptorsNew2 = np.asarray(descriptorsNew2).reshape((image2.shape[0], image2.shape[1], 128))


    #         height,width = image1.shape
    #         print("height is ", height, " width is ", width)
    #         disparity_map = np.zeros(image1.shape)

    #         for i in range(height):
    #             for j in range(0, width):
    #                 if (image1[i,j] == 0):
    #                     continue
    #                 d1_d2_dists = []
    #                 d1 = descriptorsNew1[i, j]
    #                 for k in range(0, j + 1):
    #                     d2 = descriptorsNew2[i, k]
    #                     d1_d2_dists.append(np.linalg.norm(d1 - d2))
    #                 disparity_map[i, j] = np.abs(np.argmin(d1_d2_dists) - j)
    #                 # disparity_map[(int)(y_img1)][(int)(x_img1)] = (x_img2 - x_img1)

    #         cv.imshow("my map", disparity_map)

    #         # cv.imshow("disp map", disparity_map)
    #         stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
    #         dispmap_bm = stereo_bm.compute(image1, image2)
    #         cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)

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

    
    def orbFlannCorrespondence(self, image1, image2, implementationButtonNo):
        matcher = cv.FlannBasedMatcher_create()
        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)
        
        orb = cv.ORB(1002, 1.13)

        keyPoints1, descriptors1 = orb.detectAndCompute(grayscaleImage1, None)
        keyPoints2, descriptors2 = orb.detectAndCompute(grayscaleImage2, None)

        # Creating matcher
        matches = self.matcherAndSorter(descriptors1, descriptors2)
        imageOfMatches= cv.drawMatches(grayscaleImage1, keyPoints1, grayscaleImage2, keyPoints2, matches[:100], None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("Image of Matches", imageOfMatches)
        
        # if(implementationButtonNo == 1): # OpenCV function
        #     # Stereo correspondence with block matching algorithm
        #     stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
        #     dispmap_bm = stereo_bm.compute(grayscaleImage1, grayscaleImage2)
        #     cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)
        #     cv.waitKey(0)
        # if(implementationButtonNo == 2): # My implementation
        #     pointsInImage1 = []
        #     pointsInImage2 = []
        #     for index, (m1,m2) in enumerate(matches):
        #         print("index ", index, " m1 ", m1, " m2 ", m2)
        #         if index > 2:
        #             break
        #         else:
        #             point1 = keyPoints1[m1.queryIdx].pt
        #             point2 = keyPoints2[m2.trainIdx].pt
        #             print(index, point1, point2)
        #             correspondingPoints.append([point1, point2])
        #     plt.imshow(image1, 'gray')
        #     plt.show()
        #     plt.imshow(image2, 'gray')
        #     plt.show()