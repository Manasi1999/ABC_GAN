#Function to find correlation between Discriminator Probability Vs Error 
import torch 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
import numpy as np

def discProbVsError(dataset,disc,device):
    #Discriminator Probability Vs Error 
    sample_size = len(dataset)
    test_loader = DataLoader(dataset, batch_size=sample_size, shuffle=False)
    #Check for natural variation
    errors = sample_size*[0]
    for x_batch,y_batch in test_loader: 
        y_batch = y_batch.reshape((sample_size,1))
        input = torch.cat((x_batch,y_batch),1).to(device)
        disc_pred = disc(input.float())
        disc_pred = disc_pred.reshape(1,sample_size).detach().cpu().numpy().tolist()
        plt.plot(disc_pred[0],errors,'o')
        plt.title("Discriminator Output for real data")
        plt.xlabel("Discriminator Output")
        plt.ylabel("Error (error = 0)")
        plt.show()

    err = []
    prob = []
    for i in range(5):
        sigma = 1/pow(10,i) #Different sd for gaussian error
        errors = abs(np.random.normal(0,sigma,size=(sample_size,1)))
        errors = torch.from_numpy(errors)
        for x_batch,y_batch in test_loader: 
            y_batch = y_batch.reshape((sample_size,1))
            y = y_batch + errors 
            input = torch.cat((x_batch,y),1).to(device)
            disc_pred = disc(input.float())
            disc_pred = disc_pred.detach().cpu().numpy().tolist()
            errors = errors.numpy().tolist()
            for j in range(sample_size):
                err.append(errors[j][0])
                prob.append(disc_pred[j][0])
    plt.plot(prob,err,'o')
    plt.title("Discriminator Output for noisy data")
    plt.xlabel("Discriminator Output")
    plt.ylabel("Errors")
    plt.show()
    