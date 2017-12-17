# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#load training data
trainData = pd.read_json('../input/train.json')
testData = pd.read_json('../input/test.json')

# get the max and min pixel values in trainData
maxP = -100
minP = 0
listname = ["band_1","band_2"]
for i in range(2):
    for band in trainData[listname[i]]:
        band = np.array(band)
        if( np.max(band) > maxP):
            maxP = np.max(band)
        if(np.min(band)<minP):
            minP = np.min(band)
rangeP = maxP - minP

# Process Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in trainData["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in trainData["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(trainData["is_iceberg"])
print("Xtrain:", X_train.shape)

# Process Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in testData["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in testData["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)

import matplotlib.pyplot as plt
plt.subplot(1,3,1)
plt.imshow(X_train[1,:,:,0])
plt.title('HH')
plt.subplot(1,3,2)
plt.imshow(X_train[1,:,:,1])
plt.title('HV')
plt.subplot(1,3,3)
plt.imshow(X_train[1,:,:,0]+X_train[1,:,:,1])
plt.title('HH+HV')
plt.show()

# rescale trainData where the components are between 1 and -1
X_train = X_train*(2/rangeP) - 1
# resize the train and test data to 2D
npix = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
nrow = X_train.shape[0]
Xnew = np.zeros((nrow,npix))
for i in range(nrow):
    Xnew[i,:] = X_train[i,:,:,:].reshape((1,npix))
    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
# split the dataset in two equal parts
Xtr,Xts,ytr,yts = train_test_split(Xnew,y_train,test_size=0.2,random_state=0)
# Set the parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
print("# Tuning hyper-parameters")
print()
clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(Xtr, ytr)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = yts, clf.predict(Xts)
print(classification_report(y_true, y_pred))
print()
