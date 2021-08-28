#!/usr/bin/env python3
import statsmodels.api as sm
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset , DataLoader 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statistics import mean
import pandas as pd
from sklearn import preprocessing
from dataset import CustomDataset,DataWithNoise
from network import Generator,Discriminator,GeneratorforABC 
from statsModel import statsModel 
import wandb
import hydra
from omegaconf import DictConfig
from importlib import import_module
import math
from sklearn.metrics import mean_squared_error
from statistics import mean

#Function to load functions mentioned in yaml files 
def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

#Function for ABC Pregenerator
def ABC_pre_generator(x_batch,coeff,variance,mean,device):
  coeff_len = len(coeff)
  if mean == 0:
    weights = np.random.normal(0,variance,size=(coeff_len,1))
    weights = torch.from_numpy(weights).reshape(coeff_len,1)
  else:
    weights = []
    for i in range(coeff_len):
      weights.append(np.random.normal(coeff[i],variance))
    weights = torch.tensor(weights).reshape(coeff_len,1)
  y_abc =  torch.matmul(x_batch,weights.float())
  gen_input = torch.cat((x_batch,y_abc),dim = 1).to(device)
  return gen_input 


#Function to warmup the discriminator 
def discriminator_warmup(disc,disc_opt,dataset,n_epochs,batch_size,criterion,device): 
  train_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  #Data contains elements like 
  # 1. real_data_point  1 
  # 2. fake_data_point  0
  # Fuzzy input : labels for real : 0.5 --> 1  and labels for fake : 0.5 --> 0
  # y_batch = y_batch + (1-2*y_batch)*val 
  val = 0.5
  for epoch in range(n_epochs):
    epoch_loss = 0
    for x_batch,y_batch in train_loader:
      y_batch = y_batch + (1-2*y_batch)*val
      if(val >0):
        val = val - 0.05 
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      
      disc_opt.zero_grad()

      #Train on a mixture of real and fake data 

      y_pred = disc(x_batch)
      disc_loss = criterion(y_pred, y_batch)


      # Update gradients
      disc_loss.backward(retain_graph=True)
      # Update optimizer
      disc_opt.step()

      epoch_loss += disc_loss.item()
    wandb.log({
      "Epoch(Discriminator Warm-up)":epoch,
      "disc_warmup_loss":epoch_loss/len(train_loader)
    })

def training_GAN(disc, gen,disc_opt,gen_opt,dataset, batch_size, n_epochs,criterion,coeff,mean,variance,device): 
  discriminatorLoss = []
  generatorLoss = []
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  for epoch in range(n_epochs):

    for x_batch,y_batch in train_loader:
      y_shape = list(y_batch.size()) 
      curr_batch_size = y_shape[0] 
      y_batch = torch.reshape(y_batch,(curr_batch_size,1)) 

      #Create the labels  
      real_labels = torch.ones(curr_batch_size,1).to(device)
      fake_labels = torch.zeros(curr_batch_size,1).to(device)

      #------------------------
      #Update the discriminator
      #------------------------
      disc_opt.zero_grad() 

      #Get discriminator loss for real data 
      inputs_real = torch.cat((x_batch,y_batch),dim=1).to(device) 
      disc_real_pred = disc(inputs_real)
      disc_real_loss = criterion(disc_real_pred,real_labels)

      #Get discriminator loss for fake data
      gen_input =  ABC_pre_generator(x_batch,coeff,variance,mean,device)
      generated_y = gen(gen_input)  
      x_batch = x_batch.to(device)
      inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device) 
      x_batch = x_batch.detach().cpu()
      disc_fake_pred = disc(inputs_fake) 
      disc_fake_loss = criterion(disc_fake_pred,fake_labels) 

      #Get the discriminator loss 
      disc_loss = (disc_fake_loss + disc_real_loss) / 2
      discriminatorLoss.append(disc_loss.item())

      # Update gradients
      disc_loss.backward(retain_graph=True)
      # Update optimizer
      disc_opt.step()

      #------------------------
      #Update the Generator 
      #------------------------
      gen_opt.zero_grad() 

      #Generate input to generator using ABC pre-generator 
      gen_input =  ABC_pre_generator(x_batch,coeff,variance,mean,device)
      generated_y = gen(gen_input) 
      x_batch = x_batch.to(device)
      inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device)
      x_batch = x_batch.detach().cpu()
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      generatorLoss.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()

      # wandb.log({
      # 'epoch':epoch,
      # 'gen_loss': gen_loss,
      # 'disc_real_loss': disc_real_loss,
      # 'disc_fakse_loss': disc_fake_loss,
      # 'disc_loss': disc_loss,
      # })


def test_generator(gen,dataset,coeff,mean,variance,device):
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
  mse=[]
  for epoch in range(1000):
    for x_batch, y_batch in test_loader: 
      gen_input =  ABC_pre_generator(x_batch,coeff,variance,mean,device)
      generated_y = gen(gen_input) 
      generated_y = generated_y.cpu().detach()
      generated_data = torch.reshape(generated_y,(-1,))
    gen_data = generated_data.numpy().reshape(1,len(dataset)).tolist()
    real_data = y_batch.numpy().reshape(1,len(dataset)).tolist()
    #Plot the data 
    # if(epoch%20==0):
    #   gen_data1 = generated_data.numpy().tolist()
    #   real_data1 = y_batch.numpy().tolist()
    #   plt.plot(gen_data1,'o',color='blue')
    #   plt.plot(real_data1,'o',color='red')
    #   plt.show()
    meanSquaredError = mean_squared_error(real_data,gen_data)
    mse.append(meanSquaredError)
  
  n,x,_=plt.hist(mse,bins=100,density=True)
  plt.title("Distribution of Mean Square Error ")
  sns.distplot(mse,hist=False)
  plt.show()
  print("Mean Square Error:",mean(mse))
  # wandb.log({
  #   "Mean MSE (ABC GAN)":mse_mean
  # })

  #Plot Real Vs Generated Data 
  # gen_data = generated_data.numpy().reshape(1,len(dataset)).tolist()
  # real_data = y_batch.numpy().reshape(1,len(dataset)).tolist()
  # data=[]
  # for i in range(len(dataset)):
  #     data.append([i,gen_data[0][i],"Generated"])
  #     data.append([i,real_data[0][i],"Real"])
  # table = wandb.Table(data=data, columns = ["Index", "Data","Label"])
  # wandb.log({"Real Data Vs Generated Data" : wandb.plot.scatter(table, "Index", "Data","Comparison")})

    #Weights of generator after training 
  params = torch.cat([x.view(-1) for x in gen.output.parameters()]).cpu()
  params = params.detach().numpy().tolist()
  weights = params[:-1]
  #Round to 2 decimal places 
  for i in range(len(weights)):
    weights[i] = "{:.2f}".format(weights[i])
  bias = params[len(params)-1]
  bias = "{:.2f}".format(bias)
  #print(weights)
  #print(bias)
  
def test_discriminator(disc,gen,dataset,coeff,mean,variance,device): 

  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

  for x_batch,y_batch in test_loader: 
    
    y_shape = list(y_batch.size())
    curr_batch_size = y_shape[0]
    y_batch = torch.reshape(y_batch,(curr_batch_size,1))

    #Discriminator Probability for Real Data points 
    real_data_input = torch.cat((x_batch,y_batch),dim=1).to(device)
    disc_pred = disc(real_data_input)
    disc_pred = disc_pred.detach().cpu()
    real_out = disc_pred.numpy().reshape(1,len(dataset)).tolist()
    real_out = real_out[0]
    #Discriminator Probability for Random Data Points 
    shape_data = list(real_data_input.size())
    random_data = 10*torch.rand(shape_data[0],shape_data[1]).to(device)
    disc_pred = disc(random_data)
    disc_pred = disc_pred.detach().cpu()
    rand_out = disc_pred.numpy().reshape(1,len(dataset)).tolist()
    rand_out = rand_out[0]
    #Discriminator Probability for Generated Data Points
    gen_input =  ABC_pre_generator(x_batch,coeff,variance,mean,device)
    generated_y = gen(gen_input)
    x_batch = x_batch.to(device) 
    generated_data = torch.cat((x_batch,generated_y),dim=1).to(device)
    disc_pred = disc(generated_data.float())
    disc_pred = disc_pred.detach().cpu()
    gen_out = disc_pred.numpy().reshape(1,len(dataset)).tolist()
    gen_out = gen_out[0]
    data = [[real_out[i],gen_out[i],rand_out[i]] for i in range(len(dataset))]
    wandb.log({"a_table": wandb.Table(data=data, columns=["Real ", "Generated", "Random"])})

@hydra.main(config_path="conf" ,config_name="config.yaml")
def main(cfg: DictConfig) -> None:

  #Select the device 
  cfg.train.cuda = cfg.train.cuda and torch.cuda.is_available()
  device = torch.device('cuda:0' if cfg.train.cuda else 'cpu')

  #Get the data 
  get_data = load_func(cfg.dataset.funct)
  X,Y = get_data()

  #Get stats model coefficients 
  [coeff,statsPred] = statsModel(X,Y)

  wandb.init(project='ABC-GAN', entity = 'abc-gan', config=cfg)
  run_id = wandb.run.id

  #Mean Square Error
  meanSquaredError = mean_squared_error(Y,statsPred)
  wandb.log({
    "MSE(Stats Model)":meanSquaredError,
    "Coeff (Stats Model)":coeff
  })
  
  #Get real dataset and dataset with noise 
  dataset = CustomDataset(X,Y)
  dataWithNoise = DataWithNoise(X,Y)

  #Initialize Generator and Discriminator 
  disc = Discriminator(coeff,cfg.model.hidden_nodes).to(device)
  gen = GeneratorforABC(coeff,cfg.train.initialize_generator,cfg.abc.initialize_generator_identity).to(device)

  #Add optimizer to discriminator and generator 
  optimizer = load_func(cfg.optimizer.funct)
  criterion = nn.BCEWithLogitsLoss()
  gen_opt = optimizer(gen.parameters(), lr=cfg.optimizer.lr, betas=(cfg.optimizer.beta_1,cfg.optimizer.beta_2))
  disc_opt = optimizer(disc.parameters(), lr=cfg.optimizer.lr, betas=(cfg.optimizer.beta_1,cfg.optimizer.beta_2))

  wandb.watch(disc, criterion, log="all", log_freq=10)
  wandb.watch(gen,criterion, log="all", log_freq=10)
  if(cfg.train.warmUp_discriminator==True):
    discriminator_warmup(disc,disc_opt,dataWithNoise,cfg.train.n_epochs,cfg.dataset.batch_size,criterion,device)

  #Train the GAN 
  training_GAN(disc,gen,disc_opt,gen_opt,dataset,cfg.train.n_epochs,cfg.dataset.batch_size,criterion,coeff,cfg.abc.mean,cfg.abc.variance,device)

  #Testing 
  test_generator(gen,dataset,coeff,cfg.abc.mean,cfg.abc.variance,device)
  test_discriminator(disc,gen,dataset,coeff,cfg.abc.mean,cfg.abc.variance,device)


if __name__ == "__main__":
	main()