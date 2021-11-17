from sklearn.metrics import mean_squared_error,mean_absolute_error
import scrapbook as sb 
import matplotlib.pyplot as plt
import seaborn as sns
# calculate minkowski distance
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

#Function to print performance of Stats Model 
def performance_metric(Y,Ypred):
    meanSquaredError = mean_squared_error(Y,Ypred)
    meanAbsoluteError = mean_absolute_error(Y,Ypred)

    dist1 = minkowski_distance(Y, Ypred, 1)
    dist2 = minkowski_distance(Y, Ypred, 2)

    print("Performance Metrics")
    print("Mean Squared Error:",meanSquaredError)
    print("Mean Absolute Error:",meanAbsoluteError)
    print("Manhattan distance:",dist1)
    print("Euclidean distance:",dist2)

    sb.glue("Stats Model MSE",meanSquaredError)
    sb.glue("Stats Model MAE",meanAbsoluteError)
    sb.glue("Stats Model Manhattan Distance",dist1)
    sb.glue("Stats Model Euclidean distance",dist2)

def modelAnalysis(GAN_1,ABC_GAN_1,GAN_2,ABC_GAN_2):
    #Each parameter is a array consisting of elements [mse,mae,distp1,distp2]
    params = ["MSE","MAE","Euclidean distance","Manhattan distance"]
    fig,axs = plt.subplots(4,4,figsize=(50,50))
    for i in range(4):
        #GAN_1
        axs[i,0].hist(GAN_1[i],bins=100,density=True)
        sns.distplot(GAN_1[i],hist=False,ax=axs[i,0])
        axs[i,0].set_title("GAN Model 1 - "+params[i])
        #ABC_GAN_1
        axs[i,1].hist(ABC_GAN_1[i],bins=100,density=True)
        sns.distplot(ABC_GAN_1[i],hist=False,ax=axs[i,1])
        axs[i,1].set_title("ABC GAN Model 1 - "+params[i])
        #GAN_2
        axs[i,2].hist(GAN_2[i],bins=100,density=True)
        sns.distplot(GAN_2[i],hist=False,ax=axs[i,2])
        axs[i,2].set_title("GAN Model 2 - "+params[i])
        #ABC_GAN_2
        axs[i,3].hist(ABC_GAN_2[i],bins=100,density=True)
        sns.distplot(ABC_GAN_2[i],hist=False,ax=axs[i,3])
        axs[i,3].set_title("ABC GAN Model 2 - "+params[i])

# Model Analysis for 2 Models 
def modelAnalysis2(GAN,ABC_GAN):
    #Each parameter is a array consisting of elements [mse,mae,distp1,distp2]
    params = ["MSE","MAE","Euclidean distance","Manhattan distance"]
    fig,axs = plt.subplots(4,2,figsize=(50,50))
    for i in range(4):
        #GAN_1
        axs[i,0].hist(GAN[i],bins=100,density=True)
        sns.distplot(GAN[i],hist=False,ax=axs[i,0])
        axs[i,0].set_title("GAN - "+params[i])
        #ABC_GAN_1
        axs[i,1].hist(ABC_GAN[i],bins=100,density=True)
        sns.distplot(ABC_GAN[i],hist=False,ax=axs[i,1])
        axs[i,1].set_title("ABC GAN - "+params[i])

def plotTrainingLoss(GAN_1_discLoss,GAN_1_genLoss,ABC_GAN_1_discLoss,ABC_GAN_1_genLoss,GAN_2_discLoss,GAN_2_genLoss,ABC_GAN_2_discLoss,ABC_GAN_2_genLoss):
    #Discriminator Loss 
    plt.rcParams["figure.figsize"] = [14.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    fig.suptitle('Discriminator Loss', fontsize=16)
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1,sharey=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1,sharey=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1,sharey=ax1)
    ax1.plot(GAN_1_discLoss,color='red',label = "C-GAN 1")
    ax2.plot(ABC_GAN_1_discLoss,color='blue',label = "ABC-GAN 1")
    ax3.plot(GAN_2_discLoss,color='green',label = "C-GAN 2")
    ax4.plot(ABC_GAN_2_discLoss,color='orange',label = "ABC-GAN 2")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()
    
    #Generator Loss
    plt.rcParams["figure.figsize"] = [14.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    fig.suptitle('Generator Loss', fontsize=16)
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1,sharey=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1,sharey=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1,sharey=ax1)
    ax1.plot(GAN_1_genLoss,color='red',label = "C-GAN 1")
    ax2.plot(ABC_GAN_1_genLoss,color='blue',label = "ABC-GAN 1")
    ax3.plot(GAN_2_genLoss,color='green',label = "C-GAN 2")
    ax4.plot(ABC_GAN_2_genLoss,color='orange',label = "ABC-GAN 2")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()

#Same function as above but for 2 models only 
def plotTrainingLoss2(GAN_discLoss,GAN_genLoss,ABC_GAN_discLoss,ABC_GAN_genLoss):
    #Discriminator Loss 
    plt.rcParams["figure.figsize"] = [14.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    fig.suptitle('Discriminator Loss', fontsize=16)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1,sharey=ax1)
    ax1.plot(GAN_discLoss,color='red',label = "C-GAN")
    ax2.plot(ABC_GAN_discLoss,color='blue',label = "ABC-GAN")
    ax1.legend()
    ax2.legend()
    plt.show()
    
    #Generator Loss
    plt.rcParams["figure.figsize"] = [14.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    fig.suptitle('Generator Loss', fontsize=16)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1,sharey=ax1)
    ax1.plot(GAN_genLoss,color='red',label = "C-GAN")
    ax2.plot(ABC_GAN_genLoss,color='blue',label = "ABC-GAN")
    ax1.legend()
    ax2.legend()
    plt.show()