## "Correcting Model Misspecification via Generative Adversarial Networks": [ArXiv](https://arxiv.org/abs/2304.03805)

### Getting Started 
1. Download the following dependencies
    1. jupyter 
    2. papermill 
    3. scrapbook 
    4. pytorch 
    5. seaborn 
    6. matplotlib
    7. sklearn 
    8. numpy 
    9. statistics 
    10. statsmodel 
    11. pandas 
    12. catboost 

2. Folder structure 
    1. src : contains all the functions used in the experiment. These .py files are imported into the notebooks and the functions defined here are then called in the notebook. 
        1. conf - The conf folder contains the configuration files needed and used if the experiments are run using hydra. The latest experiment flow does not use hydra and thus this can be ignored
        2. baselineModels - contains the following models along with their training and testing regime 
            1. Vanilla Neural Network 
            2. Catboost 
            3. Random Forest 
            4. Stats Model 
        3. bostonDataset, californiaDataset, cpunishDataset, diabetestDataset, friedman1Dataset, friedman2Dataset, friedman3Dataset, linearDataset, RegressionDataset imports the respective datasets and does the required preprocessing on it. 
        4. train_test.py and ABC_train_test.py contains the functions to train and test GAN and ABC GAN respectively 
        5. dataset.py : contains the Dataset class that is used to form the Dataloader (Check pytorch documentation)
        6. network.py : contains all the distriminator and generator networks used for GAN and ABC GAN 
        7. performanceMetrics.py : contains the functions to plot and compare performace metrics (MSE, MAE, Euclidean Distance,Minkowski Distance) of different models 
        8. sanityChecks : contains functions that perform sanity check by comparing discriminator probability 
    2. Notebooks1 - Contains notebooks for experiment 1 (where variance was varied)
    3. Notebooks2 - Contains notebooks for experiment 2 (where variance and bias was varied)











