


import warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from numpy import loadtxt
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
import sys
print("Enter the name of Image to predict")
path = input()


IMG_SIZE = 150
model = load_model('model1.h5')
# summarize model.
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
img = cv2.imread(path,cv2.IMREAD_COLOR)
img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
test_image = np.reshape(img,(1,IMG_SIZE,IMG_SIZE,3))
# test_image = image.load_img('dandelion.jpg', target_size = (IMG_SIZE, IMG_SIZE)) 
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = model.predict(test_image)
pred_digit = np.argmax(result,axis=1)
# print(pred_digit)



if (pred_digit == 0):
    print ('daisy')
if (pred_digit == 1):
    print ('dandelion')
if (pred_digit == 2):
    print ('rose')
if (pred_digit == 3):
    print ('sunflower')
if (pred_digit == 4):
    print ('tulip')
