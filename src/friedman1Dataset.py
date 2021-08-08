#!/usr/bin/env python3
from sklearn.datasets import make_friedman1
import statsmodels.api as sm
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def friedman1_data():
    #Load the dataset 
    X, Y = make_friedman1(n_samples=100, n_features=10, noise=0.1, random_state=None)

    #Check corelation between features and perform PCA
    corr = spearmanr(X).correlation
    print(corr)
    plt.imshow(corr)
    plt.show()

    #PCA 
    pca = PCA(n_components=6)
    pca.fit(X)
    Xp = pca.transform(X)

    #Correlation Matrix after PCA 
    corr = spearmanr(Xp).correlation
    print(corr)
    plt.imshow(corr)
    plt.show()

    #Add Constant 
    Xpc = sm.add_constant(Xp)

    return Xpc, Y 

