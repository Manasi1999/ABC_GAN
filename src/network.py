#!/usr/bin/env python3
from torch import nn

class Discriminator(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.hidden1 = nn.Linear(n_input,25)
    self.hidden2 = nn.Linear(25,50)
    self.output = nn.Linear(50,1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.hidden1(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.output(x)
    return x 

class Generator(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.hidden1 = nn.Linear(n_input,100)
    self.hidden2 = nn.Linear(100,100)
    self.output = nn.Linear(100,1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.hidden1(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.output(x)
    return x 
