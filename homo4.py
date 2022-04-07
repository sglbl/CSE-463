import cv2 as cv
import numpy as np
import scipy.interpolate


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
        print(f"Size of destImage(h and w): {destImage.shape} and height: {destImage.shape[0]}, width: {destImage.shape[1]}")
        print("Dsize of destImage is : ", destImage.shape)

        destPoints = np.array(
                        [[0, 0],
                        [destImage.shape[1]-1, 0],
                        [destImage.shape[1]-1, destImage.shape[0]-1],
                        [0, destImage.shape[0]-1]],
                     np.float32)
        
        homographyMatrix, status = cv.findHomography(srcPoints, destPoints)
        # perspectiveMatrixTransform = cv.perspectiveTransform(srcPoints[None, :, :], homographyMatrix)

        im_out = self.sgWarpPerspective(srcImage, homographyMatrix, int(destImage.shape[1]), int(destImage.shape[0])) # width x height
        # im_out = im_out.astype(np.uint8)

        # Display images
        cv.imshow("Source Image", srcImage)
        cv.imshow("Destination Image", destImage)
        cv.imshow("Warped Source Image", im_out)
        
        cv.waitKey(0)

    def to_img(self, mtr):
        V,H,C = mtr.shape
        img = np.zeros((H,V,C), dtype='uint8')
        
        # for i in range(mtr.shape[0]):
        #     # index of non-zero values
        #     y,x = np.where(mtr[i]!=0)
        #     f = scipy.interpolate.interp2d(x,y,mtr[i][mtr[i]!=0], kind='linear')
        #     X = np.arange(3)
        #     Y = np.arange(len(mtr[i]))
        #     mtr[i] = f(X,Y)
        #     # print("mtr[i is", mtr[i] )
        #     img[:,i] = mtr[i]
            
        # return img
        
        print("Wait, interpolating")
        for i in range(mtr.shape[0]): # for every row of height
            # mtr[i] is a row of the matrix which is 2d.
            y,x = np.where(mtr[i]!=0)   # If matrix[i] is not zero, then get the index of non-zero values
            xVector = np.linspace(np.min(x), np.max(x), 3) # create evenly spaced sample number of dimension(3) times.
            yVector = np.linspace(np.min(y), np.max(y), mtr.shape[1]) # Create evenly spaced sample 'width'(mtr.shape[1]) number of points
            xCoordMatrix, yCoordMatrix = np.meshgrid(xVector, yVector)   # Return coordinate matrices from coordinate vectors xx and yy
            mtr[i] = scipy.interpolate.griddata( (x,y), mtr[i][mtr[i]!=0], (xCoordMatrix, yCoordMatrix), method='nearest') 
            img[:,i] = mtr[i]   # 2d mtr[i] is a row of the image matrix

        return img

    def to_mtx(self, img):
        H,V,C = img.shape
        mtr = np.zeros((V,H,C), dtype='int')
        for i in range(img.shape[0]):
            mtr[:,i] = img[i]
        return mtr

    
    def make_interpolated_image(self, im):
        print("im.shape[0] is ",  im.shape[0])




    def sgWarpPerspective(self, srcImage, homographyMatrix, widthOfWindow, heightOfWindow):
        imageMatrix = self.to_mtx(srcImage)
        dst = np.zeros((widthOfWindow, heightOfWindow, imageMatrix.shape[2]))
        for i in range(imageMatrix.shape[0]):
            for j in range(imageMatrix.shape[1]):
                res = np.dot(homographyMatrix, [i,j,1])
                i2,j2,_ = (res / res[2] + 0.5).astype(int)
                if i2 >= 0 and i2 < widthOfWindow:
                    if j2 >= 0 and j2 < heightOfWindow:
                        dst[i2,j2] = imageMatrix[i,j]
        return self.to_img(dst)

        ################################################

        # inversed = np.linalg.inv(homographyMatrix)
        
        # print("Homography matrix is \n", homographyMatrix)
        # print("Inverse matrix is \n", inversed)
        # new_image_map = {}
        # minx, miny = widthOfWindow, heightOfWindow
        # maxx, maxy = 0, 0
        # for i in range(heightOfWindow): #height
        #     for j in range(widthOfWindow): #width
        #         xy = np.array([i, j, 1], np.float32) # x, y, 1
        #         uv = np.matmul(inversed, xy) # matrix multiplication
        #         uv = uv // uv[2]
        #         minx = min(minx, uv[0])
        #         maxx = max(maxx, uv[0])
        #         miny = min(miny, uv[1])
        #         maxy = max(maxy, uv[1])

        #         destX = (inversed[0][0]*j + inversed[0][1]*i + inversed[0][2]) / (inversed[2][0]*j + inversed[2][1]*i + inversed[2][2])
        #         destY = (inversed[1][0]*j + inversed[1][1]*i + inversed[1][2]) / (inversed[2][0]*j + inversed[2][1]*i + inversed[2][2])
                
        #         # new_image_map[j, i] = (int(destX), int(destY))
        #         # new_image_map[int(uv[0]), int(uv[1])] = (i, j)
        #         new_image_map[int(uv[0]), int(uv[1])] = (int(destY), int(destX))


        # minx, miny = int(minx), int(miny)
        # maxx, maxy = int(maxx), int(maxy)
        # print("Max x is ", maxx, " and min x is ", minx)
        # print("Max y is ", maxy, " and min y is ", miny)

        # # final_img = np.zeros((maxx - minx + 1, maxy - miny + 1), dtype=np.int) if len(srcImage.shape) == 2 else np.zeros((maxx - minx + 1, maxy - miny + 1, srcImage.shape[2]), dtype=np.int)
        # final_img = np.zeros((heightOfWindow, widthOfWindow), dtype=np.int) if len(srcImage.shape) == 2 else np.zeros((heightOfWindow, widthOfWindow, srcImage.shape[2]), dtype=np.int)

        # for k, v in new_image_map.items():
        #     # print("k is ", k , " and v is ", v)
        #     # print("src image's v position is ", srcImage[v])
        #     final_img[k[0] - minx, k[1] - miny] = srcImage[v]
        # return final_img

    def fix_image(self, img, t):
        new_image_map = {}
        minx, miny = img.shape[0], img.shape[1]
        maxx, maxy = 0, 0
        for i in range(img.shape[0]): #height
            for j in range(img.shape[1]): #width
                xy = np.array([i, j, 1], np.float32) # x, y, 1
                uv = np.matmul(t, xy) # matrix multiplication
                uv = uv / uv[2]
                minx = min(minx, uv[0])
                maxx = max(maxx, uv[0])
                miny = min(miny, uv[1])
                maxy = max(maxy, uv[1])

                new_image_map[int(uv[0]), int(uv[1])] = (i, j)

        minx, miny = int(minx), int(miny)
        maxx, maxy = int(maxx), int(maxy)
        print("Max x is ", maxx, " and min x is ", minx)
        print("Max y is ", maxy, " and min y is ", miny)

        final_img = np.zeros((maxx - minx + 1, maxy - miny + 1), dtype=np.int) if len(img.shape) == 2 else np.zeros((maxx - minx + 1, maxy - miny + 1, img.shape[2]), dtype=np.int)

        for k, v in new_image_map.items():
            final_img[k[0] - minx, k[1] - miny] = img[v]
        return final_img
