from sklearn.datasets import make_regression
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import preprocessing


def regression_data(n,m):
    #Load the dataset 
    X, Y = make_regression(n_samples=n, n_features=m, noise=0.1, random_state=None)
 
    #Creating Pandas Dataframe 
    Y = Y.reshape((Y.size,1))
    data = np.concatenate((X,Y),axis=1)

    #Create Columns 
    columns = ['X'+str(i+1) for i in range(m)]
    columns.append('Y')
    df = pd.DataFrame(data, columns = columns)
    print(df.head())

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:,0:m]
    Y = df.iloc[:,m]

    #Add Constant 
    X = X.to_numpy()
    Xc = sm.add_constant(X)
    return Xc, Y 
