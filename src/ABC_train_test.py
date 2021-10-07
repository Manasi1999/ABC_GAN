#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statistics import mean
from sklearn.metrics import mean_squared_error,mean_absolute_error

#Distance - Minkowski Function 
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

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

  #Plotting the Discriminator and Generator Loss 
  plt.plot(discriminatorLoss,color = "red",label="Discriminator Loss")
  plt.plot(generatorLoss,color="blue",label ="Generator Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
    
def test_generator(gen,dataset,coeff,w,variance,device):
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
  mse=[]
  mae=[]
  distp1 = []
  distp2 = []
  for epoch in range(1000):
    for x_batch, y_batch in test_loader: 
      gen_input =  ABC_pre_generator(x_batch,coeff,variance,w,device)
      generated_y = gen(gen_input) 
      generated_y = generated_y.cpu().detach()
      generated_data = torch.reshape(generated_y,(-1,))
    gen_data = generated_data.numpy().reshape(1,len(dataset)).tolist()
    real_data = y_batch.numpy().reshape(1,len(dataset)).tolist()
    #Plot the data 
    if(epoch%200==0):
      gen_data1 = generated_data.numpy().tolist()
      real_data1 = y_batch.numpy().tolist()
      plt.hexbin(real_data1,gen_data1,gridsize=(15,15))
      plt.xlabel("Y")
      plt.ylabel("Y_Pred")
      plt.show()
    meanSquaredError = mean_squared_error(real_data,gen_data)
    meanAbsoluteError = mean_absolute_error(real_data, gen_data)
    mse.append(meanSquaredError)
    mae.append(meanAbsoluteError)
    dist1 = minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 1)
    dist2 = minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 2)
    distp1.append(dist1)
    distp2.append(dist2)

  #Distribution of Metrics 
  #Mean Squared Error 
  n,x,_=plt.hist(mse,bins=100,density=True)
  plt.title("Distribution of Mean Square Error ")
  sns.distplot(mse,hist=False)
  plt.show()
  print("Mean Square Error:",mean(mse))

  #Mean Absolute Error 
  n,x,_=plt.hist(mae,bins=100,density=True)
  plt.title("Distribution of Mean Absolute Error ")
  sns.distplot(mae,hist=False)
  plt.show()
  print("Mean Absolute Error:",mean(mae))

  #Minkowski Distance 1st Order 
  n,x,_=plt.hist(distp1,bins=100,density=True)
  plt.title("Manhattan Distance")
  sns.distplot(distp1,hist=False)
  print("Mean Manhattan Distance:",mean(distp1))
  plt.show()
  
  #Minkowski Distance 2nd Order 
  n,x,_=plt.hist(distp2,bins=100,density=True)
  plt.title("Euclidean Distance")
  sns.distplot(distp2,hist=False)
  print("Mean Euclidean Distance:",mean(distp2))
  plt.show()
  
def test_discriminator_1(disc,gen,dataset,coeff,mean,variance,threshold,n_iterations,device): 
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
  output = [[] for i in range(len(dataset))]
  for j in range(n_iterations):
    for x_batch,y_batch in test_loader: 
      #Generate y
      gen_input = ABC_pre_generator(x_batch,coeff,variance,mean,device)
      generated_y = gen(gen_input) 
      #Get discriminator probability 
      x_batch = x_batch.to(device) 
      generated_data = torch.cat((x_batch,generated_y),dim=1).to(device)
      disc_pred = disc(generated_data.float())
      #Edits for plotting 
      disc_pred = disc_pred.detach().cpu().numpy().tolist()
      generated_y = generated_y.detach().cpu().numpy().tolist()
      for i in range(len(dataset)): 
        generated = generated_y[i][0]
        discVal = disc_pred[i][0]
        prob = 0
        #Predicted True
        if disc_pred[i][0] >=threshold:
          prob = 1
        output[i].append([generated,discVal,prob])
  
  #Plot the values
  colors = np.array(["red","green"])
  y_real = y_batch.detach().cpu().numpy().tolist()
  np_out = np.array(output) 
  for i in range(1000):
    if i%10 == 0:
      z = [int(x) for x in np_out[i][:,2]]
      plt.scatter(np_out[i][:,1],np_out[i][:,0],c=colors[z],alpha=0.2)
      plt.plot(np_out[i][:,1],np.full((100), y_real[i]),color = "k")
      plt.plot(np_out[i][:,1],np.full((100), y_real[i]+0.5))
      plt.plot(np_out[i][:,1],np.full((100), y_real[i]-0.5))
      plt.title("For datapoint "+str(i+1))
      plt.xlabel("Discriminator Output")
      plt.ylabel("Y Values")
      plt.show()
  
def test_discriminator_2(disc,gen,dataset,coeff,mean,variance,threshold,device):
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
  for x_batch,y_batch in test_loader: 
    y_real = y_batch.detach().cpu().numpy().tolist()
    #Generate y
    gen_input =  ABC_pre_generator(x_batch,coeff,variance,mean,device)
    generated_y = gen(gen_input) 
    #Get discriminator probability 
    x_batch = x_batch.to(device) 
    generated_data = torch.cat((x_batch,generated_y),dim=1).to(device)
    disc_pred = disc(generated_data.float())
    #Scatter plot 
    disc_pred = disc_pred.detach().cpu().numpy().tolist()
    generated_y = generated_y.detach().cpu().numpy().tolist()

    predTrue = []
    predFalse = []
    for i in range(len(dataset)):
      if disc_pred[i][0] >=threshold:
        predTrue.append([y_real[i],generated_y[i][0]])
      else:
        predFalse.append([y_real[i],generated_y[i][0]])
    predTrue = np.array(predTrue)
    predFalse = np.array(predFalse)
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.hexbin(predTrue[:,0],predTrue[:,1],gridsize=(15,15))
    plt.xlabel("Y real")
    plt.ylabel("Y generated")
    plt.title("Y vs Y* for datapoints predicted to be real")
    plt.subplot(122)
    plt.hexbin(predFalse[:,0],predFalse[:,1],gridsize=(15,15))
    plt.xlabel("Y real")
    plt.ylabel("Y generated")
    plt.title("Y vs Y* for datapoints predicted to be fake") 
