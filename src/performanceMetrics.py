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
