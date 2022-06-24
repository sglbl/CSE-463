import enum
from unicodedata import category
from unittest import main
import numpy as np
import cv2 as cv
import os
import tensorflow as tf
from tensorflow import keras
import random

IMAGE_SIZE = (100,100)

class Recognizer2:

    def __init__(self) -> None:
        self.trainingSet = []
        self.testSet = []
        nameOfClasses = ['banana_1', 'calculator_1', 'camera_1', 'cell_phone_1', 'flashlight_1', 'food_bag_1', 'lemon_1', 'lightbulb_1', 'lime_1', 'marker_1']
        nameLabelsOfClasses = {nameOfClass:i for i, nameOfClass in enumerate(nameOfClasses)}

        numberOfClasses = len(nameOfClasses)
        self.dataLoader(nameLabelsOfClasses)


    def dataLoader(self, nameLabelsOfClasses):
        datasetDirectory = os.path.join(os.getcwd(), "data")
        # category = ["train", "test"]

        labelsTraining = [];    labelsTest = []
        imagesTrainingAll = [];    imagesTestingAll = []
        output = []
        
        # Choose random 90% of the objects to be the training set in every class folder
        for imageFolder in os.listdir(datasetDirectory):
            label = nameLabelsOfClasses[imageFolder]
            imageFolderPath = os.path.join(datasetDirectory, imageFolder)

            ''''''''''''''' TRAINING PART '''''''''''''''
            trainingClassFolderList = os.listdir(imageFolderPath)
            # choose random 90& of the objects and get their path
            trainingClassFolderList = np.random.choice(trainingClassFolderList, int(len(trainingClassFolderList)*0.9), replace=False)
            trainingClassFolderList = [os.path.join(imageFolderPath, x) for x in trainingClassFolderList]

            # select the png files that doesn't end with depthcrop.png and maskcrop.png and get their path
            trainingClassFolderList = [x for x in trainingClassFolderList if os.path.isfile(x) 
                        and x.endswith(".png") and not x.endswith("maskcrop.png") and not x.endswith("depthcrop.png") ]         
            # read the images and resize them
            trainingImages = [cv.imread(x) for x in trainingClassFolderList]
            # convert every image to rgb
            trainingImages = [cv.cvtColor(x, cv.COLOR_BGR2RGB) for x in trainingImages]
            # resize every image to IMAGE_SIZE which is (100,100) to speed up the training
            trainingImages = [cv.resize(x, IMAGE_SIZE) for x in trainingImages]
            print("Label is " , label)
            labelsTraining.append(label)
            imagesTrainingAll.append(trainingImages)
            
            ''''''''''''''' TESTING PART '''''''''''''''
            # choose other 10% of the objects and get their path   
            testingFolderList = [x for x in trainingClassFolderList if x not in trainingClassFolderList[:int(len(trainingClassFolderList)*0.9)]]
            # select the png files that doesn't end with depthcrop.png and maskcrop.png and get their path
            testingFolderList = [x for x in testingFolderList  if os.path.isfile(x) 
                        and x.endswith(".png") and not x.endswith("depthcrop.png") and not x.endswith("maskcrop.png")]

            # read the images and resize them
            testingImages = [cv.imread(x) for x in testingFolderList]
            # convert every image to rgb
            testingImages = [cv.cvtColor(x, cv.COLOR_BGR2RGB) for x in testingImages]
            # resize every image to IMAGE_SIZE which is (100,100) to speed up the training
            testingImages = [cv.resize(x, IMAGE_SIZE) for x in testingImages]

            labelsTest.append(label)
            # imagesTestingAll.append(testingImages)
            imagesTestingAll = testingImages
            # End of for loop for every class folder
            
        # convert the lists to numpy arrays
        imagesTrainingAll = np.array(imagesTrainingAll, dtype=np.float32)
        imagesTestingAll = np.array(imagesTestingAll, dtype=np.float32)
        labelsTraining = np.array(labelsTraining, dtype=np.int32)
        labelsTest = np.array(labelsTest, dtype=np.int32)

        # Append the training and testing sets and their labels
        output.append( (imagesTrainingAll, labelsTraining) )
        output.append( (imagesTestingAll, labelsTest) )

        return output    
    
    def train(self, nameLabelsOfClasses):
        print("training")
        # Load the training and testing data
        trainingData, testingData = self.dataLoader(nameLabelsOfClasses)
        (imagesTraining, labelsTraining) = trainingData
        (imagesTesting, labelsTesting) = testingData

        # shuffle data to train the model in a better way
        imagesTraining, labelsTraining = random.shuffle(imagesTraining, labelsTraining, random_state=26)

        # create the model with Convolution Neural Network
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(nameLabelsOfClasses), activation='softmax')
        ])

        # compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # fitting model to training data this means that the model will train on the training data
        model.fit(imagesTraining, labelsTraining, epochs=10)

        # evaluate the model on the testing data
        test_loss, test_acc = model.evaluate(imagesTesting, labelsTesting)
        print('Test accuracy:', test_acc)

        # save the model
        model.save('recognitionModel.h5')
        print("model saved")

if __name__=='__main__':
    Recognizer2()


