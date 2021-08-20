#!/usr/bin/env python3

from torch.utils.data import Dataset
import numpy as np
import torch 
class CustomDataset(Dataset): 
  def __init__(self,X,y):
    self.X = X
    self.Y = y 
    self.n = len(X) #No of datapoints 
    self.p = X.shape[1] #No of features 

  def __len__(self):
    return self.n 

  def __getitem__(self,idx):
    x = self.X[idx,:]
    y = self.Y[idx]

    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)

    return (x,y)

#Creating Dataset containing of real data as well as noise (50%)
class DataWithNoise(Dataset):
  def __init__(self,X,yt):
    self.n = len(X) #Rows 
    self.p = X.shape[1] #Num features 
    self.X =  X
    self.yt = yt

  def __getitem__(self, ind):
    z = np.random.binomial(1, 0.5, 1)
    ys = z.dot(self.yt[ind]) + (1-z).dot(np.random.uniform(-0.5,0.5,1))
    Xs = self.X[ind,:]
    Xn = np.hstack((Xs,ys))

    Xn = torch.tensor(Xn,dtype=torch.float32)
    z = torch.tensor(z,dtype=torch.float32)
    return (Xn, z)

  def __len__(self):
    return self.n


