import cv2 as cv
import numpy as np       # Mathematical array operations
import scipy.interpolate # to interpolate a matrix
import tkinter as tk

class ObjectDetector:
    def __init__(self, image1, image2) -> None:
        sift = cv.SIFT_create()

        grayscaleImage1, grayscaleImage2 = self.imagesToGrayscale(image1, image2)

        keyPoint1, descriptor1 = sift.detectAndCompute(grayscaleImage1, None)
        keyPoint2, descriptor2 = sift.detectAndCompute(grayscaleImage2, None)

        # bestMatches = self.goodMatchFinder()
        # finalImage = cv.drawMatches(grayscaleImage1, keyPoint1, grayscaleImage2, keyPoint2, bestMatches, None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        matches = self.matcherAndSorter(descriptor1, descriptor2)
        finalImage= cv.drawMatches(grayscaleImage1, keyPoint1, grayscaleImage2, keyPoint2, matches[:100], None ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow("Matches", finalImage)

        cv.waitKey(0)


    def matcherAndSorter(self, descriptor1, descriptor2):
        matches = cv.BFMatcher().match(descriptor1, descriptor2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return matches

    def imagesToGrayscale(self, image1, image2):
        # Converting images to grayscale for detection
        image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        return image1, image2

    def goodMatchFinder(self, descriptor1, descriptor2):
        # Using brute for match
        matches = cv.BFMatcher().knnMatch(descriptor1, descriptor2, k=2)
        goodMatches = []
        # Find good matches by using distances
        for m,n in matches:
            if 0.70*n.distance > m.distance:
                goodMatches.append(m)
        return goodMatches
