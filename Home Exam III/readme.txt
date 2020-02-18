HOME-EXAM 3, 1MB516, HT19 
BY: Hugo Swenson, husw7546

Main scripts
------------------------------------------------------------------------------------------------------------
run_assignment is the main script for Home-exam 3.
% Task 2
The script will first build a bi-variate drug response model based on eq. (3) in the paper by Ning et al [1]
It then continues to generate a 3D-surface plot for the drug pair (AG490,U0126) on the concentration region
[0,300] × [0,100] (μM), while keeping the concentration of the third drug I-3-M at zero.
It then simulates one full experiment consisting of 512 datapoints, generating 512 response-values. 
% Task 3
The script here implements k-means clustering to generate a dataset S-train based on k-means clustering and cross validated 
feedforward networks. Through generating 30 data-points, a starting dataset is obtained, this dataset is then used to 
train network models through the function net_train. The obtained results are then used to generate the 30 values with the 
highest discrepancy through the function dicr_fun. This is then run for 5 runs (totalling a S_train with 150 datapoints)
or until a pre-defined threshold σstop = 0.05 is reached. 
% Task 4
The script here first runs Task 3 for 5 runs, disregarding the treshold and generating S_train with 150 datapoints.
This is then followed by 150 datapoints being selected randomly from the original concentration matrix, and then used
as input for net_train to generate network models. 
Finally, using the MATLAB function candexh, D-optimal design is used to select 150 row-pairs to be used for input to
net_train, generating network models. 
For each respective method, the optimal network is then selected, and used to generate 10000 response values. The mse-performance
is then evaluated for each model, and the RMSE values for each model is then calculated from these, as to see which model gives the 
predictive performance. 

Functions   
------------------------------------------------------------------------------------------------------------

hill_func represents the Hill-based global response surface model, and is ran through the input parameters X 
(the concentration matrix), and the parameter vector par_v. It yields a vector of observed values for Y.

net_train trains network models based on the input values in S_train (3 concentration vectors and one output vector)
Through a 10-fold cross validation for 6 different values for no. hidden layers, a total of 6x10 network models are 
generated. For each network, the mse is measured, and from this the rmse for each network is calculated. 
Based on the average rmse value, the best no. hidden layers is then determined. The resulting network cell, optimal
no.hidden layers, as well as the optimal value & rmse for all models are returned. 

discr_fun uses a concentration matrix, observed values generated from the concentration matrix, 
a network cell, as well as the integer value for the optimal no. hidden layers to generate the 30 values
that differ the most between the cells. The functiuon first generates 10 512 observed values from a network
based on the optimal network for each fold, with the optimal number of hidden layers.
These observed values are then sorted based on highest std-value, and the 30 values with the highest std are 
selected and returned after being matched with their respective index-position in the concentration matrix as
well as the output matrix. 

References
------------------------------------------------------------------------------------------------------------

[1] Ning S, Xu H, Al-Shyoukh I, Feng J, and Sun R. An application of a Hill-based response
surface model for a drug combination experiment in lung cancer. Statistics in Medicine,
33:4227–4236, 2014.