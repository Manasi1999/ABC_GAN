#!/usr/bin/env python3
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def energy_data():
    #Number of features 
    n_features = 8

    #Load the dataset 
    # Load the xlsx file
    data = pd.read_excel('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/ABC_GAN/ABC_GAN-2/src/energy_dataset/ENB2012_data.xlsx')
    #data = pd.read_excel('../src/energy_dataset/ENB2012_data.xlsx')
    # Read the values of the file in the dataframe
    columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']

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
    # Y =  Y.reshape((Y.size, 1))
    return X, Y 
