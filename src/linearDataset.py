from sklearn.datasets import make_regression
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Function to generate dummy real data 
def generate_real_data(n):
    X = 10*np.random.uniform(-0.5,0.5,size = n)
    m = 1
    c = 0.5
    Y = m*X + c + np.random.normal(0,1,size=n)
    X = X.reshape(n,1)
    Y = Y.reshape(n,1)
    return X, Y

def linear_data(n_samples):
    
    X,Y = generate_real_data(n_samples)
 
    #Creating Pandas Dataframe 
    data = np.concatenate((X,Y),axis=1)

    #Create Columns 
    columns = ['X','Y']
    df = pd.DataFrame(data, columns = columns)
    print(df.head())

    #Standardization of the dataset 
    scaler = preprocessing.StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:,0:1]
    Y = df.iloc[:,1]

    #Add Constant 
    X = X.to_numpy()

    #Visualization of the real data 
    plt.title("Visualizing the data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X,Y,'o',color='red')
    plt.show()

    return X, Y 
