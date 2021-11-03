from sklearn.datasets import load_diabetes
from sklearn import preprocessing
import statsmodels.api as sm
import pandas as pd
import numpy as np

def diabetes_data():
    #Number of features 
    n_features = 10 

    #Load the dataset 
    X,Y = load_diabetes(return_X_y=True)
 
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
    Xc = sm.add_constant(X)
    return Xc, Y 
