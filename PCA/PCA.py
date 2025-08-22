import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
class PCA:
    def __init__(self):
        self.filename = os.path.join(os.path.dirname(__file__), 'data', 'wine-clustering.csv')
    def read(self):
        self.df = pd.read_csv(self.filename)
    def scale_feature(self):
        self.X = (self.df-np.mean(self.df,axis=0)) / np.std(self.df,axis=0)
    def cov(self):
        self.cov_X = np.cov(self.X,rowvar=False)
        self.eig_val, self.eig_vec = np.linalg.eig(self.cov_X)
    def sort_eig(self):
        idxs = np.argsort(self.eig_val)[::-1]
        self.eig_vec = self.eig_vec[:,idxs]
        self.eig_val = self.eig_val[idxs]
    def pca(self,k):
        W = self.eig_vec[:,:k]
        self.X_PCA = np.dot(self.X,W)
    def visual1(self):
        plt.figure(figsize=(8,6))
        plt.scatter(self.X_PCA[:,0], self.X_PCA[:,1],edgecolors='k')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.show()
    def visual2(self):
        explained_var = self.eig_val / np.sum(self.eig_val)
        plt.figure(figsize=(8,6))                      
        plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
        plt.xlabel("Principal Component")              
        plt.ylabel("Variance Explained")               
        plt.title("Scree Plot")                        
        plt.show()       
pca = PCA()
pca.read()
pca.scale_feature()
pca.cov()
pca.sort_eig()
pca.pca(2)
pca.visual1()
pca.visual2()                
