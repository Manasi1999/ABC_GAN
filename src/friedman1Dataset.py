#!/usr/bin/env python3
from sklearn.datasets import make_friedman1
import statsmodels.api as sm
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def friedman1_data():
    #Load the dataset 
    X, Y = make_friedman1(n_samples=1000, n_features=10, noise=0.1, random_state=None)

    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)
    df = pd.DataFrame(data, columns = ['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y'])
    df.head()

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:,0:10]
    Y = df.iloc[:,10]

    #Check corelation between features and perform PCA
    corr = spearmanr(X).correlation
    #print(corr)
    plt.imshow(corr)
    plt.show()

    #PCA 
    pca = PCA(n_components=6)
    pca.fit(X)
    Xp = pca.transform(X)

    #Correlation Matrix after PCA 
    print("Correlation Matrix after PCA")
    corr = spearmanr(Xp).correlation
    #print(corr)
    plt.imshow(corr)
    plt.show()

    #Add Constant 
    Xpc = sm.add_constant(Xp)
    return Xpc, Y 

