from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
 
# calculate minkowski distance
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)
 
def performance_metric(Y,Ypred):
    meanSquaredError = mean_squared_error(Y,Ypred)
    meanAbsoluteError = mean_absolute_error(Y,Ypred)

    dist1 = minkowski_distance(Y, Ypred, 1)
    dist2 = minkowski_distance(Y, Ypred, 2)

    print("Performance Metrics")
    print("Mean Squared Error:",meanSquaredError)
    print("Mean Absolute Error:",meanAbsoluteError)
    print("Manhattan distance:",dist1)
    print("Euclidean distance:",dist2)