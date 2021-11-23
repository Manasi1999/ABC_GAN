import numpy as np 
import torch 
from torch import nn 
from sklearn.metrics import mean_squared_error

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
