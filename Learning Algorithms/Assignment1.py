#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:54:54 2023

@author: randyllpandohie
"""

import matplotlib.pyplot as plt
import os 
import numpy as np
import cv2 
from sklearn.decomposition import PCA, NMF, FactorAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

labels = ['Jim','Johnathan','Will']
img_size = 100
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data,dtype=object)

train = get_training_data('./data/Training')

print(train)

Jim = 0 
Johnathan = 0 
Will =0

for i, j in train:
    if j == 0:
        Jim+=1
    elif j==1:
        Johnathan+=1
    else:
        Will+=1
        
print('Jim:', Jim)
print('Johnathan:', Johnathan)
print('Will:', Will)

#plt.imshow(train[1][0], cmap='gray')
#plt.axis('off')
#print(labels[train[1][1]])

print(train.shape)

#print(train[1,0])
x=train[1,0]
print(x.shape)

pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
X=principalComponents_breast
print(principal_breast_Df.head())

print('Explained Variance = ', pca_breast.explained_variance_)
print('Principal Components = ', pca_breast.components_)

cov = np.cov(X.T)
eig_val, eig_vec = np.linalg.eig(cov)
print('Eigenvalues = ', eig_val)
print('Eigenvectors = ', eig_vec)


training_data=[]
length = len(train)

for i in range(length):
    pca = PCA(n_components=2)
    p=pca.fit_transform(train[i,0].flatten(order='F'))
   # print(p[1,:])
    training_data.append([p[0,0],p[0,1],train[i,1]])
   # print(train[i,0].flatten(order='C'))
    
    
# values of x
print(training_data)
x = np.array([1, 2, 3, 4, 5,
              6, 7, 8, 9, 10])
  
# values of y
y = np.array([10, 9, 8, 7, 6, 5,
              4, 3, 2, 1])
  
# empty list, will hold color value
# corresponding to x
col =[]
  
for i in range(0, len(x)):
    if x[i]<7:
        col.append('blue')  
    else:
        col.append('magenta') 
  
for i in range(len(x)):
      
    # plotting the corresponding x with y 
    # and respective color
    plt.scatter(x[i], y[i], c = col[i], s = 10,
                linewidth = 0)
      
  
print(plt.show())

