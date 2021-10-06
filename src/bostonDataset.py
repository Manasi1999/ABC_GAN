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
    print(boston.head())

    #Normalizing the dataset
    scaler = preprocessing.StandardScaler()
    boston = pd.DataFrame(scaler.fit_transform(boston), columns=boston.columns)
    X = boston.iloc[:, 0:13]
    y = boston.iloc[:, 13]

    # print("Preprocessing:")

    # #Check corelation between features and perform PCA
    # print("Correlation Matrix before PCA")
    # corr = spearmanr(X).correlation
    # plt.imshow(corr)
    # plt.show()

    # #PCA 
    # print("Correlation Matrix after PCA")
    # pca = PCA(n_components=6)
    # pca.fit(X)
    # Xp = pca.transform(X)

    # #Correlation Matrix after PCA 
    # corr = spearmanr(Xp).correlation
    # plt.imshow(corr)
    # plt.show()

    #Add Constant
    X = X.to_numpy() 
    Xpc = sm.add_constant(X)

    return Xpc, y 

