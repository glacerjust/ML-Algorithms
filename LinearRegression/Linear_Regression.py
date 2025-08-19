import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
class Linear_Regression:
    def __init__(self,lr=0.001,epochs=20000):
        self.filename = os.path.join(os.path.dirname(__file__), 'data', 'housing.csv')
        self.df = None
        self.X = None
        self.y = None
        self.epochs = epochs
        self.lr = lr
        self.weight = None
        self.bias = 0
    def read(self):
        self.df = pd.read_csv(self.filename)
        self.df = pd.get_dummies(self.df,columns=['ocean_proximity'])
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        self.X = self.df.drop(columns='median_house_value').values.astype(float)
        self.y = self.df['median_house_value'].values.reshape(-1, 1).astype(float)
    def scale_features(self):
        # Compute mean and std for each feature (column)
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        # Avoid division by zero for features with zero std
        self.std[self.std == 0] = 1
        # Standardize features: (x - mean) / std
        self.X = (self.X - self.mean) / self.std
    def linear(self):
        return np.dot(self.X,self.weight) + self.bias
    def error(self):
        return self.y - self.linear()
    def train(self):
        m,n = self.X.shape
        total = m
        self.weight = np.zeros((n,1))
        for i in range(self.epochs):
            dw = -2/total*np.dot(self.X.T,self.error())
            db = -2/total*np.sum(self.error())
            '''
            if i % 100 == 0:
                print(f"Epoch {i}")
                print("MSE:", np.mean(self.error()**2))
                print("Weight norm:", np.linalg.norm(self.weight))
                print("dw norm:", np.linalg.norm(dw))
                print("db:", db)
            '''
            self.weight = self.weight - self.lr*dw
            self.bias = self.bias - self.lr*db
    def predict(self):
        y_pred = self.linear()
        y_pred[y_pred < 0] = 0 #IDK why some outliner keep getting below 0
        return y_pred
    def evaluation(self):
        y_pred = self.predict()
        mse = np.mean((self.y - y_pred)**2)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)    
        return "MSE: " + str(mse)
    def visual(self):
        y_pred = self.predict()
        plt.scatter(self.y, y_pred, alpha=0.3, color='blue', label="Predictions")
        min_val = min(self.y.min(), y_pred.min())
        max_val = max(self.y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect fit")
        plt.xlabel("Actual Median House Value")
        plt.ylabel("Predicted Median House Value")
        plt.legend()
        plt.title("Actual vs Predicted House Prices")
        plt.show()
lr = Linear_Regression()
lr.read()
lr.scale_features()
lr.train()
print(lr.evaluation())
lr.visual()