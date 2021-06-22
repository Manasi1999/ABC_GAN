#!/usr/bin/env python3

import statsmodels.api as sm
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def cpunish_data():

	#Load the dataset 
	real_data = sm.datasets.cpunish.load_pandas()

	#Drop the outliers 
	real_data.data = real_data.data.drop([0,1,2])
	real_data.data = real_data.data.reset_index(drop=True)

	#Normalizing the dataset
	scaler = preprocessing.StandardScaler()
	real_data.data = pd.DataFrame(scaler.fit_transform(real_data.data), columns=real_data.data.columns)
	X = real_data.data.iloc[:, 1:7]
	y = real_data.data.iloc[:, 0]

	#Check corelation between features and perform PCA

	#Correlation Matrix before PCA 
	corr = spearmanr(X).correlation
	print(corr)
	plt.imshow(corr)
	plt.show()

	#PCA 
	pca = PCA(n_components=6)
	pca.fit(X)
	Xp = pca.transform(X)

	#Correlation Matrix after PCA 
	corr = spearmanr(Xp).correlation
	print(corr)
	plt.imshow(corr)
	plt.show()

	#Add Constant 
	Xpc = sm.add_constant(Xp)

	return Xpc, y 

