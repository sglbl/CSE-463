
    def siftBFCorrespondence2(self, image1, image2):
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

            grayscaleImage1 = cv.resize(grayscaleImage1, (int(grayscaleImage1.shape[1] / 2), int(grayscaleImage1.shape[0] / 2)))
            grayscaleImage2 = cv.resize(grayscaleImage2, (int(grayscaleImage2.shape[1] / 2), int(grayscaleImage2.shape[0] / 2)))
            image1 = grayscaleImage1
            image2 = grayscaleImage2
   
            sift = cv.SIFT_create()
            keypoints = []
            for i in range(image1.shape[0]):
                for j in range(image1.shape[1]):
                    keypoints.append(cv.KeyPoint(x=j, y=i, size=12))
            keypoints_returned, descriptorsNew1 = sift.compute(image1, keypoints)
            descriptorsNew1 = np.asarray(descriptorsNew1).reshape((image1.shape[0], image1.shape[1], 128))
            
            sift = cv.SIFT_create()
            keypoints = []
            for i in range(image2.shape[0]):
                for j in range(image2.shape[1]):
                    keypoints.append(cv.KeyPoint(x=j, y=i, size=12))
            keypoints_returned, descriptorsNew2 = sift.compute(image2, keypoints)
            descriptorsNew2 = np.asarray(descriptorsNew2).reshape((image2.shape[0], image2.shape[1], 128))


            height,width = image1.shape
            print("height is ", height, " width is ", width)
            disparity_map = np.zeros(image1.shape)

            for i in range(height):
                for j in range(0, width):
                    if (image1[i,j] == 0):
                        continue
                    d1_d2_dists = []
                    d1 = descriptorsNew1[i, j]
                    for k in range(0, j + 1):
                        d2 = descriptorsNew2[i, k]
                        d1_d2_dists.append(np.linalg.norm(d1 - d2))
                    disparity_map[i, j] = np.abs(np.argmin(d1_d2_dists) - j)
                    # disparity_map[(int)(y_img1)][(int)(x_img1)] = (x_img2 - x_img1)

            cv.imshow("my map", disparity_map)

            # cv.imshow("disp map", disparity_map)
            stereo_bm = cv.StereoBM_create(numDisparities=16, blockSize=15)
            dispmap_bm = stereo_bm.compute(image1, image2)
            cv.imshow("Disparity map from OpenCV", dispmap_bm / 255)
