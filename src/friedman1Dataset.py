#!/usr/bin/env python3
from sklearn.datasets import make_friedman1
import statsmodels.api as sm
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def friedman1_data(n_samples,n_features):
    #Load the dataset 
    X, Y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=None)

    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)
    cols = ['X'+str(i) for i in range(n_features)]
    cols.append('Y')
    df = pd.DataFrame(data, columns = cols)
    print(df.head())

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:,0:n_features]
    Y = df.iloc[:,n_features]

    #Add Constant 
    X = X.to_numpy()
    Y = Y.to_numpy()
    return X, Y 

