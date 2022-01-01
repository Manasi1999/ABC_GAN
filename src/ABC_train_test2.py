# ------------------
# This file contains all functions to train a ABC-GAN Model where the ABC pre generator can be any blackcox model
# ------------------
import torch
from torch import nn
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statistics import mean
import scrapbook as sb
from sklearn.metrics import mean_squared_error,mean_absolute_error
import network
import performanceMetrics

def ABC(prior_model,x_batch,batch_size,variance,device):
    y_abc = prior_model.predict(x_batch.numpy())
    noise = np.random.normal(0,variance, y_abc.shape)
    y_abc = y_abc + noise
    y_abc = torch.from_numpy(y_abc)
    y_abc = torch.reshape(y_abc,(batch_size,1))
    gen_input = torch.cat((x_batch,y_abc),dim = 1).float().to(device)
    return gen_input

#Training ABC_GAN for n_epochs
def training_GAN(disc,gen,disc_opt,gen_opt,dataset,batch_size,n_epochs,criterion,prior_model,variance,device): 
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
      gen_input =  ABC(prior_model,x_batch,curr_batch_size,variance,device)
      generated_y = gen(gen_input)  
      x_batch_cuda = x_batch.to(device)
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device) 
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
      gen_input =  ABC(prior_model,x_batch,curr_batch_size,variance,device)
      generated_y = gen(gen_input) 
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device)
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      generatorLoss.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()

  return discriminatorLoss,generatorLoss
    
#Training ABC_GAN until the MSE < threshold or until 5000 epochs 
def training_GAN_2(disc, gen,disc_opt,gen_opt,train_dataset,test_dataset,batch_size,error,criterion,prior_model,variance,device): 
  discriminatorLoss = []
  generatorLoss = []
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
  curr_error = error*2 
  n_epochs = 0
  while curr_error > error and n_epochs < 5000:
    n_epochs = n_epochs + 1
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
      gen_input =  ABC(prior_model,x_batch,curr_batch_size,variance,device)
      generated_y = gen(gen_input)  
      x_batch_cuda = x_batch.to(device)
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device) 
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
      gen_input =  ABC(prior_model,x_batch,curr_batch_size,variance,device)
      generated_y = gen(gen_input)
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device)
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      generatorLoss.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()

    #After every epoch check for error
    for x_batch, y_batch in test_loader: 
      gen_input =  ABC(prior_model,x_batch,len(test_dataset),variance,device)
      generated_y = gen(gen_input) 
      generated_y = generated_y.cpu().detach()
      generated_data = torch.reshape(generated_y,(-1,))
      
      gen_data = generated_data.numpy().reshape(1,len(test_dataset)).tolist()
      real_data = y_batch.numpy().reshape(1,len(test_dataset)).tolist()
      curr_error = mean_squared_error(real_data,gen_data)

  print("Number of epochs",n_epochs)
  #Store the parameters as scraps 
  sb.glue("ABC-GAN Model n_epochs",n_epochs)

  return discriminatorLoss,generatorLoss

#Training ABC-GAN Skip Connection 
#Here we need to constraint the skip connection weights between 0 and 1 after updating the generator weights 
def training_GAN_skip_connection(disc,gen,disc_opt,gen_opt,dataset, batch_size,n_epochs,criterion,prior_model,variance,device): 
  discriminatorLoss = []
  generatorLoss = []
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  constraints= network.weightConstraint()

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
      gen_input =  ABC(prior_model,x_batch,curr_batch_size,variance,device)
      generated_y = gen(gen_input)  
      x_batch_cuda = x_batch.to(device)
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device) 
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
      gen_input =  ABC(prior_model,x_batch,curr_batch_size,variance,device)
      generated_y = gen(gen_input) 
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device)
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      generatorLoss.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()

      gen._modules['skipNode'].apply(constraints)
      

  return discriminatorLoss,generatorLoss

#Testing the Model 
def test_generator(gen,dataset,prior_model,variance,expt_no,device):
  n_samples = len(dataset)
  test_loader = DataLoader(dataset,batch_size=n_samples, shuffle=False)
  mse=[]
  mae=[]
  distp1 = []
  distp2 = []
  for epoch in range(1000):
    for x_batch, y_batch in test_loader: 
      gen_input =  ABC(prior_model,x_batch,n_samples,variance,device)
      generated_y = gen(gen_input) 
      generated_y = generated_y.cpu().detach()
      generated_data = torch.reshape(generated_y,(-1,))
    
    gen_data = generated_data.numpy().reshape(1,n_samples).tolist()
    real_data = y_batch.numpy().reshape(1,n_samples).tolist()
   
    meanSquaredError = mean_squared_error(real_data,gen_data)
    meanAbsoluteError = mean_absolute_error(real_data,gen_data)
    mse.append(meanSquaredError)
    mae.append(meanAbsoluteError)
    dist1 = performanceMetrics.minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 1)
    dist2 = performanceMetrics.minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 2)
    distp1.append(dist1)
    distp2.append(dist2)

  #Storing data as scarps for analyisis via scrapbook
  sb.glue("ABC-GAN Model "+expt_no+" MSE",mean(mse))
  sb.glue("ABC-GAN Model "+expt_no+" MAE",mean(mae))
  sb.glue("ABC-GAN Model "+expt_no+" Manhattan Distance",mean(distp1))
  sb.glue("ABC-GAN Model "+expt_no+" Euclidean distance",mean(distp2))
  
  performance_metrics = [mse,mae,distp1,distp2]
  return performance_metrics