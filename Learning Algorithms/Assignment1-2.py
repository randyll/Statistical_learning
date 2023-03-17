#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:54:41 2023

@author: randyllpandohie
"""

import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA

import os
X = []
y = []
resize_val = 50

for person in os.listdir("Training"):
    files = glob.glob(os.path.join('Training', person , '*.jpg'))
    for file in files :
        img = cv2. imread ( file )
        resized_img = cv2.resize (img [:,:,0], dsize =( resize_val ,resize_val ))
        X. append ( resized_img.flatten('F'))
        y. append ( person )
        
X = np. array (X)
y = np. array (y)



Xtest = []
ytest = []
files = glob.glob(os.path.join("Testing",'*.jpg'))

for file in files :
    img = cv2.imread(file)
    resized_img = cv2.resize(img[:,:,0], dsize =( resize_val ,resize_val ))
    Xtest.append (resized_img.flatten ('F'))
    ytest.append ( file )

Xtest = np.array ( Xtest )
ytest = np.array ( ytest )



X = (X - X.mean ( axis =0)) / X.std( axis =0)
Xtest = ( Xtest - Xtest.mean ( axis =0)) / Xtest.std( axis =0)

colors = ['red', 'green', 'blue']
person_to_color = dict (zip (np.unique(y), colors ))

ytest_person = ['Will', 'Will', 'Johnathan', 'Johnathan','Jim', 'Jim']
#print(ytest)
#for _yt in ytest:
 #   for _y in np.unique(y):
  #      if _y in _yt:
   #         ytest_person.append(_y)
        #    break


pca = PCA()
pcs = pca.fit_transform(X)
pcs_test = pca.transform(Xtest)
print(pcs_test)
kaccuracy_dict = {}
for k in [2]:
    if k == 'all':
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(pcs, y)
        ytestpred = knn.predict(pcs_test)
        print(ytestpred)
        print(ytest_person)
        kaccuracy_dict[k] = accuracy_score(ytest_person,ytestpred)
    else:
        knn = KNeighborsClassifier(n_neighbors =4)
        knn.fit(pcs[:,:k], y)
        ytestpred = knn.predict(pcs_test[:,:k])
        print(ytestpred)
        print(ytest_person)
        kaccuracy_dict[k]= accuracy_score(ytest_person,ytestpred)
        
print('pca')
print(kaccuracy_dict)
print(ytest)

fig =plt.figure( figsize =(12 ,8))
for i, _y in enumerate (np.unique (y)):
    plt.scatter (pcs[y==_y ,0], pcs[y==_y ,1], label =_y , color =colors[i], marker ='o')
    for k,c in person_to_color.items():
        if k in _y:
            color = c
            break
    if '1' in _y:
        marker = 'x'
    else :
        marker ='+'
    plt.scatter(pcs_test[ytest ==_y ,0], pcs_test [ ytest ==_y ,1],label =_y , color =color ,marker =marker , s=200)
    color = None

plt.legend(loc=(1.04 , 0.5))
plt.tight_layout()
plt.savefig('pca_space.png')
plt.close()


kpca = KernelPCA ( kernel = 'cosine')  
pcs = kpca.fit_transform (X)
pcs_test = kpca.transform ( Xtest )
knn = KNeighborsClassifier ( n_neighbors =4)
knn.fit(pcs, y)
ytestpred = knn. predict ( pcs_test )
kaccuracy_dict = accuracy_score ( ytest_person, ytestpred )

print(kaccuracy_dict)
      
fig =plt.figure( figsize =(12 ,8))
for i, _y in enumerate (np.unique (y)):
    plt.scatter (pcs[y==_y ,0], pcs[y==_y ,1], label =_y , color =colors[i], marker ='o')
    for k,c in person_to_color.items():
        if k in _y:
            color = c
            break
    if '1' in _y:
        marker = 'x'
    else :
        marker ='+'
    plt.scatter(pcs_test[ytest ==_y ,0], pcs_test [ ytest ==_y ,1],label =_y , color =color ,marker =marker , s=200)
    color = None

plt.legend(loc=(1.04 , 0.5))
plt.tight_layout()
plt.savefig('pca_space_l.png')
plt.close()