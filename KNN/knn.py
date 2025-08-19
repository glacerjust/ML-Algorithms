import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os
class KNN:
    def __init__(self,k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    def euclidean_distance(self,a,b):
        return np.sqrt(np.sum((a - b)**2)) #we use np.linalg.norm(self.X_train-x,axis=1) instead so this function is for intuition behind the distance
    def _predict(self,x):
        distances = np.linalg.norm(self.X_train-x,axis=1)
        k_idx = np.argsort(distances)[:self.k]
        k_neighbors = self.y_train[k_idx]
        most_common = Counter(k_neighbors).most_common(1)
        return most_common[0][0]
    def predicts(self,X_test):
        return np.array([self._predict(x) for x in X_test])
    def score(self,X_test,y_test):
        predictions = self.predicts(X_test)
        return np.mean(predictions == y_test)
knn = KNN()
filename = os.path.join(os.path.dirname(__file__), 'data', 'breastcancerdata.csv')
df = pd.read_csv(filename)
df.dropna(axis=1, how="all")

X = df.drop(columns=['id','diagnosis']).values

'''
#z-score
# axis = 0 for the mean and std of each column 
mean = np.mean(X,axis=0) 
std = np.std(X,axis=0) 
X = (X-mean)/std
'''
"""
#MinMaxScalar
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X = (X - X_min) / np.where(X_max - X_min == 0, 1, X_max - X_min)
"""
y = df['diagnosis'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))
plt.scatter(X_train[y_train == 'B',0],X_train[y_train == 'B',1],color='red',label='Benign')
plt.scatter(X_train[y_train == 'M',0],X_train[y_train == 'M',1],color='green',label='Malignant')
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.legend()
plt.show()