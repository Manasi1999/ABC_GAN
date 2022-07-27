import numpy as np 
import torch 
from torch import nn 
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import catboost as ctb
import scrapbook as sb 
from performanceMetrics import performance_metric
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scrapbook as sb 
from pytorch_tabnet.tab_model import TabNetRegressor

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

class NeuralNetwork(torch.nn.Module):
    def __init__(self,n_input,n_output):
        super().__init__()
        self.hidden = nn.Linear(n_input,100)
        self.output = nn.Linear(100,n_output)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x 

# This function will fit a vanilla neural network on the dataset provided and return the MSE values 
def vanillaNeuralNetwork(train_dataset,test_dataset,batch_size,n_features,n_target,n_epochs):
    #DataLoader 
    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
    test_iter = torch.utils.data.DataLoader(test_dataset,batch_size = len(test_dataset),shuffle = True)

    #Initialize the Network 

    net = NeuralNetwork(n_features,n_target)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)

    #Training 
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_iter:
            y_pred = net(x_batch)
            loss = criterion(y_pred,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #     print("epoch {} loss: {:.4f}".format(epoch + 1, loss.item()))
    # print("TRAINING COMPLETE")


    #Testing 
    for x_test,y_test in test_iter:
        y_test = torch.reshape(y_test,(len(test_dataset),n_target))
        y_pred = net(x_test)
        y_test = y_test.detach().cpu().numpy().reshape(n_target,len(test_dataset)).tolist()
        y_pred = y_pred.detach().cpu().numpy().reshape(n_target,len(test_dataset)).tolist()
        mse = mean_squared_error(y_pred,y_test)
        sb.glue("Vanilla NN MSE", mse)
        print("Mean Squared error",mse)

# This function will fit a Random Forest Regressor on the dataset and return the MSE values 
def randomForest(X_train,y_train,X_test,y_test):

    #Training 
    regr = RandomForestRegressor(max_depth=4, random_state=42)
    regr.fit(X_train, y_train)

    #Testing 
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    print("Mean Squared error",mse)
    mae = mean_absolute_error(y_pred,y_test)
    return mae

# This function will fit catboost on the dataset and return the MSE values 
def catboost(X_train,y_train,X_test,y_test):

    #Training
    model_CB = ctb.CatBoostRegressor()
    model_CB.fit(X_train, y_train)
    #print(model_CBC)

    #Testing
    y_pred = model_CB.predict(X_test)
    mae = mean_absolute_error(y_pred,y_test)
    print("Mean Absolute error",mae)

    return mae

def catboost2(X_train,y_train,X_test,y_test,variance):

    #Training
    model_CB = ctb.CatBoostRegressor()
    model_CB.fit(X_train, y_train)
    #print(model_CBC)

    #Testing
    y_pred = model_CB.predict(X_test)
    mae = mean_absolute_error(y_pred+np.random.normal(0,variance, y_pred.shape),y_test)
    print("Mean Absolute error",mae)

    return mae

def tabnetreg(x_train,y_train,x_test,y_test,batch_size,n_features,n_target,n_epochs, lr):
    clf = TabNetRegressor(optimizer_fn=torch.optim.Adam, 
                        optimizer_params=dict(lr = 0.001),
                        mask_type= 'sparsemax',
                        verbose = 1)  
  
    clf.fit(X_train = x_train,y_train = y_train, 
        eval_set=[(x_train, y_train), (X_test, y_test)],
        eval_name=['train', 'valid'], 
        eval_metric=[ 'mae'], 
        max_epochs = n_epochs, 
        batch_size = batch_size,
        patience=50)

    preds = clf.predict(x_test)

    y_true = y_test

    test_score = mean_absolute_error(y_pred=preds, y_true=y_true)

    print(f"BEST VALID SCORE FOR dataset : {clf.best_cost}")
    print(f"FINAL TEST SCORE FOR dataset: {test_score}")
    print(clf.history)

    # plot losses
    #plt.figure(feat)
    plt.plot(clf.history['loss'])

    return  test_score
