#!/usr/bin/env python3

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

	
	plt.plot(Y,'o',color='red',label = 'Real')
	plt.plot(ypred,'o',color='blue',label = 'Predicted')
	plt.title("Y predicted and Y real")
	plt.legend()
	plt.show()
	
	return coefficients,ypred


