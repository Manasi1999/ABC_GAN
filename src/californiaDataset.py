#!/usr/bin/env python3
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm
from sklearn import preprocessing
import numpy as np 
import pandas as pd


def california_data():
    #Number of features 
    n_features = 8 

    #Load the dataset 
    X,Y = fetch_california_housing(return_X_y=True)
 
    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)

    #Create Columns 
    columns = ['X'+str(i+1) for i in range(n_features)]
    columns.append('Y')
    df = pd.DataFrame(data, columns = columns)
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

