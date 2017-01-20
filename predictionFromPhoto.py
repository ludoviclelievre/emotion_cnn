#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:48:50 2016

@author: ludoviclelievre
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2
import pickle
from sklearn.externals import joblib
import os

#DATA_PATH = os.environ['EMOTION_PROJECT']
DATA_PATH = "/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/donnees/fer2013"
#GIT_PATH = "C:\Users\KD5299\Python-Project"
GIT_PATH = "/Users/ludoviclelievre/Documents/Python-Project"

###########
class ImgFromPhoto(Data):
    
    def __init__(self,img):
        self.data_emotion = np.array([-1])
        self.data_usage = np.array(['FromPhoto'])
        self.data_image = img
    
    def NormalizationExternalData(self):
        # set the image norm to 100 
        self.data_image = min_max_scaler.transform(self.data_image)
        # use the scaler from the training set to standardize each pixel of the image
        self.data_image = normalization_scaler.transform(self.data_image)


# load xml
cascPath = os.path.join(GIT_PATH,'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascPath)

# load model
model = load_model(os.path.join(GIT_PATH,'modelflip48'))
normalization_scaler = joblib.load(os.path.join(GIT_PATH,'normalization_scaler'))
min_max_scaler = joblib.load(os.path.join(GIT_PATH,'min_max_scaler'))

# new dictionnary
dico_emotion = {0:'Angry', 1:'Fear',
                2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral',-1:'?'}
                
# function predicting emotion from a photo: input = the path of the photo
def predict_emotion(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print('The path of you image is not correct')
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img_gray = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(img_gray, (48,48), interpolation = cv2.INTER_AREA)
            expand_img = np.expand_dims(resized_img.ravel(), axis=0)
            image=ImgFromPhoto(expand_img)
            image.SubstractMean()
            image.NormalizationExternalData()
            X, _ = image.CreateUsageSet('FromPhoto') 
            img_pred = model.predict_classes(X, batch_size=128, verbose=0)
            # show the photo with the predicted class
            plt.title('it seems you are '+str(dico_emotion[img_pred[0]]))
            plt.axis('off')
            plt.imshow(img_gray, cmap='gray')

# enter an image path
image_path = '...'
predict_emotion(image_path)





