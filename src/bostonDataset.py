#!/usr/bin/env python3
from sklearn.datasets import load_boston
import statsmodels.api as sm
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def boston_data():
    #Load the dataset 
    boston_dataset = load_boston()

    #Create Pandas Dataframe 
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['TARGET'] = boston_dataset.target


    #Normalizing the dataset
    scaler = preprocessing.StandardScaler()
    boston = pd.DataFrame(scaler.fit_transform(boston), columns=boston.columns)
    X = boston.iloc[:, 0:13]
    y = boston.iloc[:, 13]

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

    return Xpc, y 

