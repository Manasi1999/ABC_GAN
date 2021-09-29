#!/usr/bin/env python3
from sklearn.datasets import make_friedman2
import statsmodels.api as sm
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def friedman2_data():
    #Load the dataset 
    X, Y = make_friedman2(n_samples=1000, noise=0.1, random_state=None)

    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)
    df = pd.DataFrame(data, columns = ['X1','X2','X3','X4','X4','Y'])
    df.head()

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:,0:4]
    Y = df.iloc[:,4]

    #Add Constant 
    X = X.to_numpy()
    Xc = sm.add_constant(X)
    return Xc, Y 

