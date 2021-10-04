#!/usr/bin/env python3
from performanceMetrics import performance_metric
import statsmodels.api as sm
import matplotlib.pyplot as plt 

def statsModel(X,Y):
 
	model = sm.OLS(Y,X)

	res = model.fit()
	print(res.summary())

	print('Parameters: ', res.params)

	#Store the coefficients for ABC Pre Generator 
	coefficients  = [res.params[i] for i in range(res.params.size)]

	#Prediction using stats Model 
	ypred = res.predict(X)
	
	plt.hexbin(Y,ypred,gridsize=(15,15))
	plt.title("Y_real vs Y_predicted")
	plt.xlabel("y_real")
	plt.ylabel("y_predicted")
	plt.legend()
	plt.show()
	
	performance_metric(Y,ypred)

	return coefficients,ypred


