# ------------------
# This file contains all functions to train a ABC-GAN Model where the ABC pre generator 
# is a custom defined model 
# ------------------
import torch
from torch.utils.data import DataLoader 
import numpy as np
from statistics import mean, variance , stdev
import scrapbook as sb
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import network
import performanceMetrics

def linear_model(X) :
    Y = 4 * X[:,0] + 3 * X[:,1] + 12 * X[:,2] + 2 * X[:,3] + X[:,4] + 10 
    return Y

def quadratic_model(X) : 
    Y = 10 * np.sin(math.pi * X[:,0]* X[:,1]) + 20 * np.square(X[:,2]+0.5) + 10 * X[:,3] + 5* X[:,4]
    mu = mean(Y) 
    sd = stdev(Y)
    Y = np.subtract(Y,mu)
    Y = np.divide(Y,sd)
    return Y

def pre_generator(X,prior_model,variance,batch_size,device):
    Y = prior_model.predict(X.numpy())
    if type(variance) == int or type(variance) == float :
        Y = Y + np.random.normal(0,variance,Y.shape)
    else:
        variance = np.square(X[:,1]) + 2 * abs(X[:,2])
        Y = Y + np.random.normal(0,variance)
    Y = torch.from_numpy(Y)
    Y = torch.reshape(Y,(batch_size,1))
    gen_input = torch.cat((X,Y),dim = 1).float().to(device)
    return gen_input

def training_GAN(disc,gen,disc_opt,gen_opt,dataset,batch_size,n_epochs,criterion,prior_model,variance,device): 
  discriminatorLoss = []
  generatorLoss = []

  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  for epoch in range(n_epochs):
    epoch_loss_disc = []
    epoch_loss_gen = []
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
      gen_input = pre_generator(x_batch,prior_model,variance,curr_batch_size,device)
      generated_y = gen(gen_input)  
      x_batch_cuda = x_batch.to(device)
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device) 
      disc_fake_pred = disc(inputs_fake) 
      disc_fake_loss = criterion(disc_fake_pred,fake_labels) 

      #Get the discriminator loss 
      disc_loss = (disc_fake_loss + disc_real_loss) / 2
      epoch_loss_disc.append(disc_loss.item())

      # Update gradients
      disc_loss.backward(retain_graph=True)
      # Update optimizer
      disc_opt.step()

      #------------------------
      #Update the Generator 
      #------------------------
      gen_opt.zero_grad() 

      #Generate input to generator using ABC pre-generator 
      gen_input =  pre_generator(x_batch,prior_model,variance,curr_batch_size,device)
      generated_y = gen(gen_input) 
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device)
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      epoch_loss_gen.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()
    
    discriminatorLoss.append(sum(epoch_loss_disc)/len(epoch_loss_disc))
    generatorLoss.append(sum(epoch_loss_gen)/len(epoch_loss_gen))

  performanceMetrics.plotTrainingLoss2(discriminatorLoss,generatorLoss,np.linspace(1, n_epochs, n_epochs).astype(int))

#Training ABC-GAN Skip Connection 
#Here we need to constraint the skip connection weights between 0 and 1 after updating the generator weights 
def training_GAN_skip_connection(disc,gen,disc_opt,gen_opt,dataset,batch_size,n_epochs,criterion,prior_model,variance,device): 
  discriminatorLoss = []
  generatorLoss = []
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  constraints= network.weightConstraint()

  for epoch in range(n_epochs):
    epoch_loss_disc = []
    epoch_loss_gen = []
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
      gen_input =  pre_generator(x_batch,prior_model,variance,curr_batch_size,device)
      generated_y = gen(gen_input)  
      x_batch_cuda = x_batch.to(device)
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device) 
      disc_fake_pred = disc(inputs_fake) 
      disc_fake_loss = criterion(disc_fake_pred,fake_labels) 

      #Get the discriminator loss 
      disc_loss = (disc_fake_loss + disc_real_loss) / 2
      epoch_loss_disc.append(disc_loss.item())

      # Update gradients
      disc_loss.backward(retain_graph=True)
      # Update optimizer
      disc_opt.step()

      #------------------------
      #Update the Generator 
      #------------------------
      gen_opt.zero_grad() 

      #Generate input to generator using ABC pre-generator 
      gen_input =  pre_generator(x_batch,prior_model,variance,curr_batch_size,device)
      generated_y = gen(gen_input) 
      inputs_fake = torch.cat((x_batch_cuda,generated_y),dim=1).to(device)
      disc_fake_pred = disc(inputs_fake)

      gen_loss = criterion(disc_fake_pred,real_labels)
      epoch_loss_gen.append(gen_loss.item())

      #Update gradients 
      gen_loss.backward()
      #Update optimizer 
      gen_opt.step()

      gen._modules['skipNode'].apply(constraints)

    discriminatorLoss.append(sum(epoch_loss_disc)/len(epoch_loss_disc))
    generatorLoss.append(sum(epoch_loss_gen)/len(epoch_loss_gen))

  performanceMetrics.plotTrainingLoss2(discriminatorLoss,generatorLoss,np.linspace(1, n_epochs, n_epochs).astype(int))

#Testing the Model 
def test_generator(gen,dataset,prior_model,variance,expt_no,device):
  n_samples = len(dataset)
  test_loader = DataLoader(dataset,batch_size=n_samples, shuffle=False)
  mse=[]
  mae=[]
  distp1 = []
  distp2 = []
  for epoch in range(100):
    for x_batch, y_batch in test_loader: 
      gen_input =  pre_generator(x_batch,prior_model,variance,n_samples,device)
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
  