#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statistics import mean
import pandas as pd
import scrapbook as sb 
from sklearn.metrics import mean_squared_error,mean_absolute_error

#Distance - Minkowski Function 
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

#Function to warm-up the discriminator 
def discriminator_warmup(disc,disc_opt,dataset,n_epochs,batch_size,criterion,device): 
  train_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

#Training GAN 1 - This function trains the nexwork(GAN)for n_epochs
def training_GAN(disc, gen,disc_opt,gen_opt,dataset, batch_size, n_epochs,criterion,device): 
    discriminatorLoss = []
    generatorLoss = []
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(n_epochs):
        for x_batch,y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_shape = list(y_batch.size()) 
            curr_batch_size = y_shape[0] 
            y_batch = torch.reshape(y_batch,(curr_batch_size,1)).to(device) 

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
            z= np.random.normal(0,1,size=(curr_batch_size,1))
            z = torch.from_numpy(z).to(device)
            gen_input = torch.cat((x_batch,z),dim=1).to(device) 
            generated_y = gen(gen_input.float()).to(device)  
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
            z= np.random.normal(0,1,size=(curr_batch_size,1))
            z = torch.from_numpy(z).to(device)
            gen_input = torch.cat((x_batch,z),dim=1).to(device) 
            #Generate input to generator using ABC pre-generator 
            generated_y = gen(gen_input.float()).to(device) 
            inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device)
            disc_fake_pred = disc(inputs_fake)

            gen_loss = criterion(disc_fake_pred,real_labels)
            generatorLoss.append(gen_loss.item())

            #Update gradients 
            gen_loss.backward()
            #Update optimizer 
            gen_opt.step()
    return discriminatorLoss,generatorLoss

#Training GAN 2 - This function trains the nexwork(GAN) until the mse < error or 30,000 epochs have passed
def training_GAN_2(disc,gen,disc_opt,gen_opt,train_dataset,test_dataset,batch_size,error,criterion,device):
  discriminatorLoss = []
  generatorLoss = []
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
  curr_error = error*2 
  n_epochs = 0
  while curr_error > error and n_epochs < 5000:
      n_epochs = n_epochs + 1 
      for x_batch,y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_shape = list(y_batch.size()) 
        curr_batch_size = y_shape[0] 
        y_batch = torch.reshape(y_batch,(curr_batch_size,1)).to(device) 

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
        z= np.random.normal(0,1,size=(curr_batch_size,1))
        z = torch.from_numpy(z).to(device)
        gen_input = torch.cat((x_batch,z),dim=1).to(device) 
        generated_y = gen(gen_input.float()).to(device)  
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
        z= np.random.normal(0,1,size=(curr_batch_size,1))
        z = torch.from_numpy(z).to(device)
        gen_input = torch.cat((x_batch,z),dim=1).to(device) 
        #Generate input to generator using ABC pre-generator 
        generated_y = gen(gen_input.float()).to(device) 
        inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device)
        disc_fake_pred = disc(inputs_fake)

        gen_loss = criterion(disc_fake_pred,real_labels)
        generatorLoss.append(gen_loss.item())

        #Update gradients 
        gen_loss.backward()
        #Update optimizer 
        gen_opt.step()
        
      #After every epoch check the error 
      for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device) 
        y_batch = y_batch.to(device)
        z= np.random.normal(0,1,size=(len(test_dataset),1))
        z = torch.from_numpy(z).to(device)
        gen_input = torch.cat((x_batch,z),dim=1).to(device) 
        generated_y = gen(gen_input.float()).to(device) 
        generated_y = generated_y.cpu().detach()
        generated_data = torch.reshape(generated_y,(-1,))

        gen_data = generated_data.detach().cpu().numpy().reshape(1,len(test_dataset)).tolist()
        real_data = y_batch.detach().cpu().numpy().reshape(1,len(test_dataset)).tolist()
        curr_error = mean_squared_error(real_data,gen_data)

  print("Number of epochs needed",n_epochs)
  sb.glue("GAN Model n_epochs",n_epochs)
  
  return discriminatorLoss,generatorLoss

def training_GAN_3(disc,gen,disc_opt,gen_opt,dataset,batch_size,t_loss,criterion,device):
  discriminatorLoss = []
  generatorLoss = []
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  curr_loss = t_loss*2
  n_epochs = 0
  while curr_loss > t_loss and n_epochs < 5000:
      n_epochs = n_epochs + 1 
      for x_batch,y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_shape = list(y_batch.size()) 
        curr_batch_size = y_shape[0] 
        y_batch = torch.reshape(y_batch,(curr_batch_size,1)).to(device) 

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
        z= np.random.normal(0,1,size=(curr_batch_size,1))
        z = torch.from_numpy(z).to(device)
        gen_input = torch.cat((x_batch,z),dim=1).to(device) 
        generated_y = gen(gen_input.float()).to(device)  
        inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device) 

        disc_fake_pred = disc(inputs_fake) 
        disc_fake_loss = criterion(disc_fake_pred,fake_labels) 

        #Get the discriminator loss 
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        curr_loss = disc_loss
        discriminatorLoss.append(disc_loss.item())

        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        #------------------------
        #Update the Generator 
        #------------------------
        gen_opt.zero_grad() 
        z= np.random.normal(0,1,size=(curr_batch_size,1))
        z = torch.from_numpy(z).to(device)
        gen_input = torch.cat((x_batch,z),dim=1).to(device) 
        #Generate input to generator using ABC pre-generator 
        generated_y = gen(gen_input.float()).to(device) 
        inputs_fake = torch.cat((x_batch,generated_y),dim=1).to(device)
        disc_fake_pred = disc(inputs_fake)

        gen_loss = criterion(disc_fake_pred,real_labels)
        generatorLoss.append(gen_loss.item())

        #Update gradients 
        gen_loss.backward()
        #Update optimizer 
        gen_opt.step()
    

  print("Number of epochs needed",n_epochs)
  sb.glue("GAN Model n_epochs",n_epochs)
  
  return discriminatorLoss,generatorLoss


#Testing the Generator - After 1st training 
def test_generator(gen,dataset,device):
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
  mse=[]
  mae=[]
  distp1 = []
  distp2 = []
  for epoch in range(1000):
    for x_batch, y_batch in test_loader: 
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      z= np.random.normal(0,1,size=(len(dataset),1))
      z = torch.from_numpy(z).to(device)
      gen_input = torch.cat((x_batch,z),dim=1).to(device) 
      generated_y = gen(gen_input.float()).to(device) 
      generated_y = generated_y.cpu().detach()
      generated_data = torch.reshape(generated_y,(-1,))

    gen_data = generated_data.detach().cpu().numpy().reshape(1,len(dataset)).tolist()
    real_data = y_batch.detach().cpu().numpy().reshape(1,len(dataset)).tolist()

    #Performance Metrics 
    meanSquaredError = mean_squared_error(real_data,gen_data)
    meanAbsoluteError = mean_absolute_error(real_data, gen_data)
    dist1 = minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 1)
    dist2 = minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 2)
    mse.append(meanSquaredError)
    mae.append(meanAbsoluteError)
    distp1.append(dist1)
    distp2.append(dist2)
    
  #Distribution of Metrics 
  #Mean Squared Error 
  # n,x,_=plt.hist(mse,bins=100,density=True)
  # plt.title("Distribution of Mean Square Error ")
  # sns.distplot(mse,hist=False)
  # plt.show()
  # print("Mean Square Error:",mean(mse))

  # #Mean Absolute Error 
  # n,x,_=plt.hist(mae,bins=100,density=True)
  # plt.title("Distribution of Mean Absolute Error ")
  # sns.distplot(mae,hist=False)
  # plt.show()
  # print("Mean Absolute Error:",mean(mae))

  # #Minkowski Distance 1st Order 
  # n,x,_=plt.hist(distp1,bins=100,density=True)
  # plt.title("Manhattan Distance")
  # sns.distplot(distp1,hist=False)
  # plt.show()
  # print("Mean Manhattan Distance:",mean(distp1))
  

  # #Minkowski Distance 2nd Order 
  # n,x,_=plt.hist(distp2,bins=100,density=True)
  # plt.title("Euclidean Distance")
  # sns.distplot(distp2,hist=False)
  # plt.show()
  # print("Mean Euclidean Distance:",mean(distp2))

  sb.glue("GAN Model MSE",mean(mse))
  sb.glue("GAN Model MAE",mean(mae))
  sb.glue("GAN Model Manhattan Distance",mean(distp1))
  sb.glue("GAN Model Euclidean distance",mean(distp2))
  performanceMetrics = [mse,mae,distp1,distp2]
  return performanceMetrics

#Testing the Generator - After 2nd training 
def test_generator_2(gen,dataset,device):
  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
  mse=[]
  mae=[]
  distp1 = []
  distp2 = []
  for epoch in range(1000):
    for x_batch, y_batch in test_loader: 
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      z= np.random.normal(0,1,size=(len(dataset),1))
      z = torch.from_numpy(z).to(device)
      gen_input = torch.cat((x_batch,z),dim=1).to(device) 
      generated_y = gen(gen_input.float()).to(device) 
      generated_y = generated_y.cpu().detach()
      generated_data = torch.reshape(generated_y,(-1,))

    gen_data = generated_data.detach().cpu().numpy().reshape(1,len(dataset)).tolist()
    real_data = y_batch.detach().cpu().numpy().reshape(1,len(dataset)).tolist()

    #Performance Metrics 
    meanSquaredError = mean_squared_error(real_data,gen_data)
    meanAbsoluteError = mean_absolute_error(real_data, gen_data)
    dist1 = minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 1)
    dist2 = minkowski_distance(np.array(real_data)[0],np.array(gen_data)[0], 2)
    mse.append(meanSquaredError)
    mae.append(meanAbsoluteError)
    distp1.append(dist1)
    distp2.append(dist2)
    
  #Distribution of Metrics 
  #Mean Squared Error 
  # n,x,_=plt.hist(mse,bins=100,density=True)
  # plt.title("Distribution of Mean Square Error ")
  # sns.distplot(mse,hist=False)
  # plt.show()
  # print("Mean Square Error:",mean(mse))

  # #Mean Absolute Error 
  # n,x,_=plt.hist(mae,bins=100,density=True)
  # plt.title("Distribution of Mean Absolute Error ")
  # sns.distplot(mae,hist=False)
  # plt.show()
  # print("Mean Absolute Error:",mean(mae))

  # #Minkowski Distance 1st Order 
  # n,x,_=plt.hist(distp1,bins=100,density=True)
  # plt.title("Manhattan Distance")
  # sns.distplot(distp1,hist=False)
  # plt.show()
  # print("Mean Manhattan Distance:",mean(distp1))
  
  # #Minkowski Distance 2nd Order 
  # n,x,_=plt.hist(distp2,bins=100,density=True)
  # plt.title("Euclidean Distance")
  # sns.distplot(distp2,hist=False)
  # plt.show()
  # print("Mean Euclidean Distance:",mean(distp2))

  sb.glue("GAN Model 2 MSE",mean(mse))
  sb.glue("GAN Model 2 MAE",mean(mae))
  sb.glue("GAN Model 2 Manhattan Distance",mean(distp1))
  sb.glue("GAN Model 2 Euclidean distance",mean(distp2))
  performanceMetrics = [mse,mae,distp1,distp2]
  return performanceMetrics
   
def test_discriminator(disc,gen,dataset,device): 

  test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

  for x_batch,y_batch in test_loader: 

    y_shape = list(y_batch.size())
    curr_batch_size = y_shape[0]
    y_batch = torch.reshape(y_batch,(curr_batch_size,1))
    
    #Real Data Points  
    real_data_input = torch.cat((x_batch,y_batch),dim=1).to(device)
    disc_pred = disc(real_data_input)
    disc_pred = disc_pred.detach().cpu()
    real_out = disc_pred.numpy().reshape(1,len(dataset)).tolist()
    real_out = real_out[0]
    #Random Data Points 
    shape_data = list(real_data_input.size())
    random_data = 10*torch.rand(shape_data[0],shape_data[1]).to(device)
    disc_pred = disc(random_data)
    disc_pred = disc_pred.detach().cpu()
    rand_out = disc_pred.numpy().reshape(1,len(dataset)).tolist()
    rand_out = rand_out[0]
    #Generated Data points 
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    generated_out = gen(x_batch)
    generated_data = torch.cat((x_batch,generated_out),dim=1).to(device)
    disc_pred = disc(generated_data.float())
    disc_pred = disc_pred.detach().cpu()
    gen_out = disc_pred.numpy().reshape(1,len(dataset)).tolist()
    gen_out = gen_out[0]
    data = [[real_out[i],gen_out[i],rand_out[i]] for i in range(len(dataset))]


