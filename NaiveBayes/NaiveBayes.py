from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
class NaiveBayes:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_priors = None
        self.feature_params = None  #store mean/var for numeric, counts for categorical        
        self.filename = os.path.join(os.path.dirname(__file__), 'data', 'adult.csv')
    def read(self):
        df = pd.read_csv(self.filename)
        df['is-USA'] = (df['native-country'] == 'United-States').astype(int)
        df['has-capital-gain'] = (df['capital-gain']>0).astype(int)
        df['has-capital-loss'] = (df['capital-loss']>0).astype(int)
        df['normalized-age'] = (df['age'] - np.mean(df['age'])) / np.std(df['age'])
        df = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation','relationship','race'])
        df['gender'] = (df['gender'] == 'Male').astype(int)
        df = df.drop(columns=['fnlwgt','education','age','native-country'])
        X = df.drop('income',axis=1).astype(np.float64).values
        y = (df['income']== '>50K').astype(int).values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def parameter(self):
        n_samples, n_features = self.X_train.shape
        self._classes = np.unique(self.y_train)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self._var = np.zeros((n_classes,n_features),dtype=np.float64)
        self._priors = np.zeros(n_classes,dtype=np.float64)

        for i_class, c in enumerate(self._classes):
            X_c = self.X_train[self.y_train == c]
            self._mean[i_class,:] = X_c.mean(axis=0)
            self._var[i_class,:] = X_c.var(axis=0)
            self._priors[i_class] = X_c.shape[0] / float(n_samples)
    def predict(self):
        y_pred = np.array([self._predict(x) for x in self.X_test])
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print(f"Accuracy : {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall   : {rec*100:.2f}%")
        print(f"F1-score : {f1*100:.2f}%")
        return
    def _pdf(self,class_idx,x):
        epsilon = 1e-6
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        return np.maximum((np.exp(-((x-mean))**2 / (var*2 + epsilon))) / (np.sqrt(2* np.pi * (var+epsilon))),1e-10)
    def _predict(self,x):
        posteriors = []
        for class_idx, c in enumerate(self._classes):
            prior = np.log(self._priors[class_idx])
            posterior = np.sum(np.log(self._pdf(class_idx,x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)] #return class with highest chance 
nb = NaiveBayes()
nb.read()
nb.parameter()
nb.predict()
