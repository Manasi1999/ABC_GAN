import numpy as np 
import torch 
from torch import nn 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import catboost as ctb

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

    return mse

# This function will fit catboost on the dataset and return the MSE values 
def catboost(X_train,y_train,X_test,y_test):

    #Training
    model_CBC = ctb.CatBoostRegressor()
    model_CBC.fit(X_train, y_train)
    #print(model_CBC)

    #Testing
    y_pred = model_CBR.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    print("Mean Squared error",mse)

    return mse