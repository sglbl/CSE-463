import numpy as np
import cv2 as cv
import os
import pandas as pd
import sklearn
import sklearn.cluster
import sklearn.metrics
import sklearn.preprocessing

class Recognizer:
    def __init__(self) -> None:
        print("hello")
        self.objects = ['banana_1', 'calculator_1', 'camera_1', 'cell_phone_1', 'flashlight_1', 'food_bag_1', 'lemon_1', 'lightbulb_1', 'lime_1', 'marker_1']
        # self.bananaSet = []; self.calculatorSet = []; self.cameraSet = []; self.cellPhoneSet = []; self.flashlightSet = []; 
        # self.foodBagSet = []; self.lemonSet = []; self.lightbulbSet = []; self.limeSet = []; self.markerSet = []
        for objectFolderName in self.objects:
            self.objectFolderPath = os.path.join(os.getcwd(), "data", objectFolderName)
            self.objectFolderList = os.listdir(self.objectFolderPath)
            self.objectFolderList.sort()
            self.objectFolderList = [os.path.join(self.objectFolderPath, x) for x in self.objectFolderList]
            self.objectFolderList = [x for x in self.objectFolderList if os.path.isfile(x) and x.endswith(".png")]
            self.objectFolderList = [cv.imread(x) for x in self.objectFolderList]
            # Choose random 90% of the objects to be the training set
            self.trainingSet = np.random.choice(len(self.objectFolderList), int(len(self.objectFolderList)*0.9), replace=False)
            self.trainingSet = [self.objectFolderList[x] for x in self.trainingSet]
            print(self.trainingSet)
            # Learning every object in the folder using sift
            self.trainingSet = [cv.SIFT_create().detectAndCompute(x, None) for x in self.trainingSet]
            # Choose the remaining 10% of the objects to be the testing set
            self.testingSet = [x for x in range(len(self.objectFolderList)) if x not in self.trainingSet]
            self.testingSet = [self.objectFolderList[x] for x in self.testingSet]
            self.testingSet = [cv.SIFT_create().detectAndCompute(x, None) for x in self.testingSet]
            
            # Use features with svm to train
            self.svm = svm.SVC()
            self.svm.fit([x[0] for x in self.trainingSet], [y for y in range(len(self.trainingSet))])
            # Use features with svm to test
            self.predictions = self.svm.predict([x[0] for x in self.testingSet])
            # Compare the predictions with the testing set
            self.correct = [x for x in range(len(self.testingSet)) if x == self.predictions[x]]
            # Print the accuracy of the svm
            print(objectFolderName, len(self.correct)/len(self.testingSet))
            
            

    
    def goodMatchFinder(self, descriptor1, descriptor2):
        matches = cv.BFMatcher().knnMatch(descriptor1, descriptor2, k=2)
        goodMatches = []
        # Find good matches by using distances
        for m,n in matches:
            if 0.70*n.distance > m.distance:
                goodMatches.append(m)
        return goodMatches       


if __name__=='__main__':
    Recognizer()

