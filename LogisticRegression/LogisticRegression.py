import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
class Logistic_Regression:
    def __init__(self,lr=0.001,epochs=5000):
        self.filename = os.path.join(os.path.dirname(__file__), 'data', 'breastcancerdata.csv')
        self.df = None
        self.X = None
        self.y = None
        self.epochs = epochs
        self.lr = lr
        self.weight = None
        self.bias = 0
        self.loss_history = []
    def scale_features(self):
        self.mean = np.mean(self.X,axis=0)
        self.std = np.std(self.X,axis=0)
        self.std[self.std == 0] = 1 #edge case where std = 0
        self.X = (self.X - self.mean)/self.std
    def read(self):
        self.df = pd.read_csv(self.filename)
        self.X = self.df.drop(columns=['id','diagnosis']).values
        self.y = self.df['diagnosis'].map({'M': 1, 'B': 0}).values.reshape(-1,1)
    def sigmoid(self,z):
        return 1/(1+ np.exp(-z))
    def train(self):
        total = len(self.y)
        m,n = self.X.shape 
        self.weight = np.zeros((n,1))
        for i in range(self.epochs):
            z = np.dot(self.X,self.weight) + self.bias
            sig = self.sigmoid(z)
            loss = -1/total * np.sum(self.y*np.log(sig + 1e-8) + (1 - self.y)*np.log(1 - sig + 1e-8))
            self.loss_history.append(loss)
            dCdw = 1/total*np.dot(self.X.T,(sig-self.y))
            dCdb = 1/total*np.sum(sig-self.y)
            self.weight = self.weight - self.lr*dCdw
            self.bias = self.bias - self.lr*dCdb
    def predict(self):
        z = np.dot(self.X,self.weight) + self.bias
        g = self.sigmoid(z)
        predictions = (g >= 0.5).astype(int)
        accuracy = np.mean(predictions == self.y)
        print(accuracy)
        return predictions
    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(range(self.epochs), self.loss_history, color='blue')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def plot_predictions(self, predictions):
        plt.figure(figsize=(6, 4))
        plt.scatter(range(len(self.y)), self.y, label='Actual', marker='o', alpha=0.7)
        plt.scatter(range(len(predictions)), predictions, label='Predicted', marker='x', alpha=0.7)
        plt.title('Predicted vs Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.legend()
        plt.show()

data = Logistic_Regression()
data.read()
data.scale_features()
data.train()
preds = data.predict()
data.plot_loss()
data.plot_predictions(preds)