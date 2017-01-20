# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:45:53 2016

@author: KD5299
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:16:40 2016

@author: ludoviclelievre
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
import os
from skimage import exposure,transform
import pandas as pd
from sklearn import preprocessing
import pickle
#import getpass
#getpass.getuser()
# Importation des donnees

np.random.seed(1337)  # for reproducibility

#DATA_PATH = os.environ['EMOTION_PROJECT']
DATA_PATH = "/Users/ludoviclelievre/Documents/cours_ensae_ms/python_pour_le_dataScientist/projet_python/donnees/fer2013"
#DATA_PATH = "mypath"
#GIT_PATH = "C:\Users\KD5299\Python-Project"
GIT_PATH = "/Users/ludoviclelievre/Documents/Python-Project"

df0 = pandas.read_csv(os.path.join(DATA_PATH,'fer2013.csv'), 
                     sep=",")
df0.drop('pixels',axis = 1,inplace=True)
df1 = pandas.read_csv(os.path.join(DATA_PATH,'pixels.csv'), 
                             sep=" ", header=None)

df = pd.merge(df0,df1,left_index=True,right_index=True)
# dico emotion
dico_emotion = {0:'Angry', 1:'Fear',
                2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}

class Data:
    def __init__(self,df):
        self.data_emotion = df['emotion'].as_matrix(columns=None)
        self.data_usage = df['Usage'].as_matrix(columns=None)
        self.data_image = df[list(filter(lambda pxl: type(pxl)!=str ,df.columns.tolist()))].as_matrix(columns=None)
   
    @property
    def nb_example(self):
        return int(self.data_emotion.shape[0])
    @property
    def dim(self):
        return int(np.sqrt(self.data_image[0].shape[0]))     
    @property
    def nb_classes(self):
        return int(np.unique(self.data_emotion).shape[0])
    @property
    def input_shape(self):
        if K.image_dim_ordering() == 'th':
            return (1, self.dim, self.dim)
        else:
            return (self.dim, self.dim,1)
    
    def CreateUsageSet(self,usage):
        mask = np.in1d(self.data_usage, usage)
        X = self.data_image[mask, :]
        Y = self.data_emotion[mask]
    
        if K.image_dim_ordering() == 'th':
            X = X.reshape(X.shape[0], 1,self.dim, self.dim)
        else:
            X = X.reshape(X.shape[0], self.dim, self.dim, 1)
        X = X.astype('float32')
        Y = np_utils.to_categorical(Y, self.nb_classes)
        return X,Y

    def zoom(self,z):
        data_image_zoom = np.ndarray((self.data_image.shape[0],
                                      self.data_image.shape[1]/z**2))
        i = 0
        for image in self.data_image:
            data_image_zoom[i] = transform.downscale_local_mean(
                            image.reshape((self.dim, self.dim)),(z,z)).ravel()
            i=1+i
        self.data_image = data_image_zoom    
#        self.dim = int(self.dim / z)
        
    def EnhanceContrast(self):
        self.data_image = np.apply_along_axis(
                                exposure.equalize_hist,1,self.data_image)
        
    def Normalize(self):
        self.data_image =self.data_image/255.

    def ViewEmotion(self):
        fig = plt.figure()
        i = 1
       
        nrow = int(np.sqrt(self.nb_example+.25)-0.5)+1
        for emotion,image in zip(self.data_emotion,self.data_image):
            ax = fig.add_subplot(nrow,nrow+1,i)
            pixels = image.reshape(self.input_shape[0:2])
            ax.imshow(pixels, cmap='gray')
            ax.set_title(dico_emotion[emotion])
            plt.axis('off')
            i = i+1
            
    def ViewOneEmotion(self,example):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image=self.data_image[example]
        emotion = self.data_emotion[example]
        pixels = image.reshape(self.input_shape[0:2])
        ax.imshow(pixels, cmap='gray')
        ax.set_title(dico_emotion[emotion])
        plt.axis('off')
    
    # Substract the mean value of each image
    def SubstractMean(self):
        mean = self.data_image.mean(axis=1)
        self.data_image = self.data_image - mean[:, np.newaxis]

    # set the image norm to 100 and standardized each pixels accross the image    
    def Normalization(self):
        # set the image norm to 100 
        #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
        #self.data_image = min_max_scaler.fit_transform(self.data_image)
        self.data_image = preprocessing.normalize(self.data_image, norm='l2')*10
        # standardized each pixels accross the image
        scaler = preprocessing.StandardScaler().fit(self.data_image[self.data_usage=='Training'])
        self.data_image = scaler.transform(self.data_image)

    def FlipTrain(self,usage):
        flip_image = self.data_image[self.data_usage==usage]*0
        i = 0
        for image in self.data_image[self.data_usage==usage]:
            flip_image[i] = np.fliplr(
                    image.reshape(self.input_shape[0:2])).ravel()
            i=1+i
        flip_emotion = self.data_emotion[self.data_usage==usage]
        flip_usage = self.data_usage[self.data_usage==usage]+" flip"

        self.data_image = np.concatenate(
                    (self.data_image,flip_image),axis=0)
        self.data_emotion = np.concatenate(
                    (self.data_emotion,flip_emotion),axis=0)     
        self.data_usage = np.concatenate(
                    (self.data_usage,flip_usage),axis=0)
      
#==============================================================================
# CNN
#==============================================================================
data = Data(df[df['emotion']!=1])
f = lambda x: x-1 if x>1 else x
fv = np.vectorize(f)
data.data_emotion = fv(data.data_emotion)
print("flip")
data.FlipTrain('Training') # create 'Training flip'
data.SubstractMean()
data.Normalization()
data.zoom(2)
data.nb_example
# set inputs and outputs
Xtrain, YtrainBin = data.CreateUsageSet(['Training','Training flip']) # add 'Training flip'
Xcv, YcvBin = data.CreateUsageSet('PublicTest')
Xtest, YtestBin = data.CreateUsageSet('PrivateTest')

### parameters CNN ###
batch_size = 128
nb_epoch = 32
# input image dimensions
img_rows, img_cols = data.dim,data.dim
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# CNN model
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=data.input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(data.nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Model fitting
history24 = model.fit(Xtrain, YtrainBin, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(Xcv, YcvBin))
# save
model.save(os.path.join(GIT_PATH,'modelflip48'))
pickle.dump(history24.history,open( os.path.join(GIT_PATH,'historyflip48'), "wb" ))


