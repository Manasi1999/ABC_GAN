#!/usr/bin/env python3
from performanceMetrics import performance_metric
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scrapbook as sb 

def statsModel(X_train,Y_train,X_test,Y_test):
	model = sm.OLS(Y_train,X_train)

	res = model.fit()
	print(res.summary())

	#Store the coefficients for ABC Pre Generator 
	coefficients  = [res.params[i] for i in range(res.params.size)]

	#Prediction using stats Model 
	ypred = res.predict(X_test)
	
	plt.hexbin(Y_test,ypred,gridsize=(15,15))
	plt.title("Y_real vs Y_predicted")
	plt.xlabel("y_real")
	plt.ylabel("y_predicted")
	plt.legend()
	plt.show()
	
	performance_metric(Y_test,ypred)
	return coefficients,ypred


