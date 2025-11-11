import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
class SVM:
    def __init__(self,lr=0.001,epochs=1000,lamda=0.01):
        self.filename = os.path.join(os.path.dirname(__file__), 'data', 'breastcancerdata.csv')
        self.lr = lr
        self.epochs = epochs
        self.weight = None
        self.bias = 0
        self.df = None
        self.x = None
        self.y = None
        self.lamda = lamda
    def scale_features(self):
        # Compute mean and std for each feature (column)
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        # Avoid division by zero for features with zero std
        self.std[self.std == 0] = 1
        # Standardize features: (x - mean) / std
        self.x = (self.x - self.mean) / self.std
    def readfile(self):
        self.df = pd.read_csv(self.filename)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed|^id', case=False)]
        self.df['diagnosis'] = self.df['diagnosis'].map({'M':1, 'B':-1})
        self.y = self.df['diagnosis'].values
        self.x = self.df.drop(columns='diagnosis').values
        self.scale_features()
    def train(self):
        m,n = self.x.shape
        self.weight = np.zeros(n)
        for _ in range(self.epochs):
            for i in range(m):
                condition = self.y[i]*(np.dot(self.x[i],self.weight) -self.bias)
                if float(condition) >= 1:
                    dw = 2*self.lamda*self.weight
                    db = 0
                else:
                    dw = 2*self.lamda*self.weight - self.y[i]*self.x[i]
                    db = self.y[i]
                self.weight = self.weight - self.lr * dw
                self.bias = self.bias - self.lr * db
    def predict(self):
        prediction = np.dot(self.x,self.weight)-self.bias
        return np.where(prediction >=0,1,-1)
    def accuracy(self):
        pred = self.predict()
        accuracy = np.mean(pred == self.y)
        return accuracy
    def plot_decision_boundary(self, feature1_idx=0, feature2_idx=1):
        """
        Plots the decision boundary using two features.
        feature1_idx, feature2_idx: indices of the features to plot
        """
        x1 = self.x[:, feature1_idx]
        x2 = self.x[:, feature2_idx]

        # Scatter plot of actual classes
        plt.figure(figsize=(6,5))
        plt.scatter(x1[self.y==1], x2[self.y==1], color='r', label='Malignant')
        plt.scatter(x1[self.y==-1], x2[self.y==-1], color='b', label='Benign')

        # Create a grid to plot decision boundary
        x1_range = np.linspace(min(x1), max(x1), 50)
        x2_range = -(self.weight[feature1_idx]*x1_range - self.bias) / self.weight[feature2_idx]

        plt.plot(x1_range, x2_range, color='k', linestyle='--', label='Decision Boundary')
        plt.xlabel(f'Feature {feature1_idx}')
        plt.ylabel(f'Feature {feature2_idx}')
        plt.legend()
        plt.title('SVM Decision Boundary')
        plt.show()
    
    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix comparing predicted vs actual labels.
        """
        pred = self.predict()
        cm = confusion_matrix(self.y, pred, labels=[1, -1])  # Keep label order consistent
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant (1)', 'Benign (-1)'])
        
        plt.figure(figsize=(5, 4))
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()
svm = SVM()
svm.readfile()
svm.train()
print(svm.accuracy())

# Plot decision boundary for first two features
svm.plot_decision_boundary(feature1_idx=0, feature2_idx=1)

# Plot predicted vs actual
svm.plot_confusion_matrix()