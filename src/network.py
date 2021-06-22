#!/usr/bin/env python3

from torch import nn

#coefficient = [c,w_1,w_2,...,w_n]
#y = w_1*x_1 + w_2*x_2 + ... w_n*x_n + c 

class Discriminator(nn.Module):

  def __init__(self,coefficients,hiddenNodes):

    super().__init__()
    inputNodes = len(coefficients)+1
    self.hidden = nn.Linear(inputNodes,hiddenNodes)
    self.output = nn.Linear(hiddenNodes,1)
    #Define LeakyRelu Activation 
    self.relu = nn.ReLU()

  def forward(self, x):
    #Pass the input tensor through the operations 
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return x 


class Generator(nn.Module):

  def __init__(self,coefficients,initialize):
    super().__init__()
    inputNodes = len(coefficients)
    #Input to Output Layer Linear Transformation
    self.output = nn.Linear(inputNodes,1)
    if initialize == True:
      self.initialize_weights(coefficients)

  def forward(self, x):
    #Pass the input tensor through the operations 
    x = self.output(x)
    return x 

  def initialize_weights(self,coefficients):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        for i in range(1,len(coefficients)):
          nn.init.constant_(m.weight[0][i-1],coefficients[i])
        nn.init.constant_(m.bias,coefficients[0])
