#!/usr/bin/env python3
from torch import nn
import torch 
import math 
import pytorch_tabnet
import pytorch_tabnet.tab_network

class Discriminator(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.hidden1 = nn.Linear(n_input,25)
    self.hidden2 = nn.Linear(25,50)
    self.output = nn.Linear(50,1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.hidden1(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x) 
    x = self.output(x)
    x = self.sigmoid(x)
    return x 

class Generator(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.hidden1 = nn.Linear(n_input,50)
    self.hidden2 = nn.Linear(50,50)
    self.output = nn.Linear(50,1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.hidden1(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.output(x)
    return x 

#SKIP CONNECTION 
#Custom Function to write skip connection 

class skipConnection(nn.Module):

    def __init__(self, in_features = 2, out_features = 1, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #Only 1 weight as a parameter 
        self.weight = torch.nn.Parameter(torch.Tensor(1,1))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    #Initialise the weights  
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  
        
    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = torch.Tensor(x,1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(x):
          output[i] = input[i][0]*self.weight + input[i][1]*(1-self.weight)
        # weights = torch.tensor([self.weight,1-self.weight])
        # output = input.matmul(weights.t())
        if self.bias is not None:
            output += self.bias
        ret = output
        ret = ret.to(device)
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class GeneratorWithSkipConnection(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.hidden1 = nn.Linear(n_input,50)
    self.hidden2 = nn.Linear(50,50)
    self.output = nn.Linear(50,1)
    self.skipNode = skipConnection()
    self.relu = nn.ReLU()

  def forward(self, x):
    y_abc = x[:,-1] 
    samples = y_abc.size(dim=0)
    y_abc = torch.reshape(y_abc,(samples,1))
    x = self.hidden1(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    y_gan = self.output(x)
    out = torch.cat((y_gan , y_abc),1)
    out = self.skipNode(out)
    return out 

#Module to constraint weights - contraint the weights of the skip connection layer to be between 0 and 1 
class weightConstraint(object):
    def __init__(self):
        pass   
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(0,1)
            module.weight.data=w


# generator of GAN as tabnet
class GeneratorTabnet(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.output = pytorch_tabnet.tab_network.TabNet(n_input, 1)

  def forward(self, x):
    x = self.output(x)
    return x[0]
    


#generator of skipGAN as tabnet
class GeneratorTabnetskipConnection(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.output = pytorch_tabnet.tab_network.TabNet(n_input, 1)
    self.skipNode = skipConnection()

  def forward(self, x):
    y_abc = x[:,-1] 
    samples = y_abc.size(dim=0)
    y_abc = torch.reshape(y_abc,(samples,1))
    y_gan = self.output(x)
    out = torch.cat((y_gan[0] , y_abc),1)
    out = self.skipNode(out)
    return out 

class DiscriminatorTabnet(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    # self.hidden1 = nn.Linear(n_input,25)
    # self.hidden2 = nn.Linear(25,50)
    self.output = pytorch_tabnet.tab_network.TabNet(n_input, 1)
    # self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # x = self.hidden1(x)
    # x = self.relu(x)
    # x = self.hidden2(x)
    # x = self.relu(x) 
    x = self.output(x)
    x = self.sigmoid(x[0])
    return x 
  

#SKIP CONNECTION 
#Custom Function to write skip connection 

class modifiedSkipConnection(nn.Module):
    
    def __init__(self, in_features = 2, out_features = 1, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #Only 1 weight as a parameter 
        self.weight = torch.nn.Parameter(torch.Tensor(1,1))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    #Initialise the weights  
    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.weight,0)
        
    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = torch.Tensor(x,1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        w = torch.sigmoid(self.weight)
        for i in range(x):
          output[i] = input[i][0]*w + input[i][1]*(1-w)
        if self.bias is not None:
            output += self.bias
        ret = output
        ret = ret.to(device)
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class GeneratorWithModifiedSkipConnection(nn.Module):
  def __init__(self,n_input):
    super().__init__()
    self.hidden1 = nn.Linear(n_input,50)
    self.hidden2 = nn.Linear(50,50)
    self.output = nn.Linear(50,1)
    self.skipNode = modifiedSkipConnection()
    self.relu = nn.ReLU()

  def forward(self, x):
    y_abc = x[:,-1] 
    samples = y_abc.size(dim=0)
    y_abc = torch.reshape(y_abc,(samples,1))
    x = self.hidden1(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu(x)
    y_gan = self.output(x)
    out = torch.cat((y_gan , y_abc),1)
    out = self.skipNode(out)
    return out 