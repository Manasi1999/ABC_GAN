#!/usr/bin/env python3
from sklearn.datasets import load_boston
import statsmodels.api as sm
from sklearn import preprocessing
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def boston_data():
    #Number of features 
    n_features = 13 

    #Load the dataset 
    X,Y = load_boston(return_X_y=True)
 
    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)

    #Create Columns 
    columns = ['X'+str(i+1) for i in range(n_features)]
    columns.append('Y')
    df = pd.DataFrame(data, columns = columns)

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print(df.head())
    X = df.iloc[:,0:n_features]
    Y = df.iloc[:,n_features]

    #Add Constant 
    X = X.to_numpy()

    return X, Y 

