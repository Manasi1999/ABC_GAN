#!/usr/bin/env python3
from sklearn.datasets import make_friedman3
import statsmodels.api as sm
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def friedman3_data(n_samples):
    #Load the dataset 
    X, Y = make_friedman3(n_samples=n_samples, noise=0.1, random_state=None)

    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)
    df = pd.DataFrame(data, columns = ['X1','X2','X3','X4','Y'])
    print(df.head())

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:,0:4]
    Y = df.iloc[:,4]

    #Add Constant 
    X = X.to_numpy()
    Y = Y.to_numpy()
    return X, Y 

