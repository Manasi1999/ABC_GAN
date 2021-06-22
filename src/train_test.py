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
from network import Generator,Discriminator 
from statsModel import statsModel 
import wandb
import hydra
from omegaconf import DictConfig
from importlib import import_module
import math

#Function to load functions mentioned in yaml files 
def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

#Function to warmup the discriminator 
def discriminator_warmup(disc,disc_opt,dataset,n_epochs,batch_size,criterion,device): 
  train_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  for epoch in range(n_epochs):
    epoch_loss = 0
    for x_batch,y_batch in train_loader:
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

    print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

def training_GAN(disc, gen,disc_opt,gen_opt,dataset, batch_size, n_epochs,criterion,device): 
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
      x_batch = x_batch.to(device) 
      generated_y = gen(x_batch)  
      inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device) 

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
      generated_y = gen(x_batch) 
      inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device)
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      generatorLoss.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()

      wandb.log({
      'epoch':epoch,
      'gen_loss': gen_loss,
      'disc_real_loss': disc_real_loss,
      'disc_fakse_loss': disc_fake_loss,
      'disc_loss': disc_loss,
      })

  #Plotting the Discriminator and Generator Loss 
  plt.plot(discriminatorLoss,color = "red",label="Discriminator Loss")
  plt.plot(generatorLoss,color="blue",label ="Generator Loss")
  plt.legend()
  plt.show()

def test_generator(gen,dataset,device):
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
  for x_batch, y_batch in test_loader: 
    x_batch = x_batch.to(device)
    generated_y = gen(x_batch)
    generated_y = generated_y.cpu().detach()
    #Plot Real and Generated Data 
    generated_data = torch.reshape(generated_y,(-1,))
    print(generated_data)
    print(y_batch)

  ## TODO : Plots 
  
  
  
  #Weights of generator after training 
  params = torch.cat([x.view(-1) for x in gen.output.parameters()]).cpu()
  params = params.detach().numpy().tolist()
  weights = params[:-1]
  #Round to 2 decimal places 
  for i in range(len(weights)):
    weights[i] = "{:.2f}".format(weights[i])
  bias = params[len(params)-1]
  bias = "{:.2f}".format(bias)
  print("Generator Weights after Training")
  print("Weights : ",weights)
  print("Bias : ",bias)

def test_discriminator(disc,gen,dataset,device): 

  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

  for x_batch,y_batch in test_loader: 

    y_shape = list(y_batch.size())
    curr_batch_size = y_shape[0]
    y_batch = torch.reshape(y_batch,(curr_batch_size,1))
    print("For real data points")
    real_data_input = torch.cat((x_batch,y_batch),dim=1).to(device)
    disc_pred = disc(real_data_input)
    print(disc_pred)

    print("For random data points")
    shape_data = list(real_data_input.size())
    random_data = 10*torch.rand(shape_data[0],shape_data[1]).to(device)
    disc_pred = disc(random_data)
    print(disc_pred) 

    print("For generated data points")
    x_batch = x_batch.to(device)
    generated_out = gen(x_batch)
    generated_data = torch.cat((x_batch,generated_out),dim=1).to(device)
    disc_pred = disc(generated_data.float())
    print(disc_pred)


@hydra.main(config_path="conf" ,config_name="config.yaml")
def main(cfg: DictConfig) -> None:

  #Select the device 
  cfg.train.cuda = cfg.train.cuda and torch.cuda.is_available()
  device = torch.device('cuda:0' if cfg.train.cuda else 'cpu')

  #Get the data 
  get_data = load_func(cfg.dataset.funct)
  X,Y = get_data()

  #Get stats model coefficients 
  coeff = statsModel(X,Y)

  wandb.init(project='ABC-GAN', entity = 'abc-gan', config=cfg)
  run_id = wandb.run.id

  #Get real dataset and dataset with noise 
  dataset = CustomDataset(X,Y)
  dataWithNoise = DataWithNoise(X,Y)

  #Initialize Generator and Discriminator 
  disc = Discriminator(coeff,cfg.model.hidden_nodes).to(device)
  gen = Generator(coeff,cfg.train.initialize_generator).to(device)

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
  training_GAN(disc,gen,disc_opt,gen_opt,dataset,cfg.train.n_epochs,cfg.dataset.batch_size,criterion,device)

  #Testing 
  test_generator(gen,dataset,device)
  test_discriminator(disc,gen,dataset,device)


if __name__ == "__main__":
	main()