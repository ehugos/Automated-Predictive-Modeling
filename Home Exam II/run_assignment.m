clear all; 
close all; 

% Runscript for home-exam 2, automated predictive modeling, Created by 
% Hugo Swenson in collaboration with Erik Berner-Wik

%% Task 1

% Sets the dimension d to 2000
d = 2000; 

% Generates a 500 dimensional parameter vector 
w_1 = repmat(1/(d/4),d/4,1);
% Generates a 1500 dimesnional vector and adds it to the 500 dim vector
w_1 = [w_1 ; zeros((3/4)*d,1)];

w_2 = repmat(1/(d/200),d/200,1);
w_2 = [w_2 ; zeros(d-10,1)]; 

% Generates the vector w3
for i = 1:d
    w_3(i,1) = 2^(-i); 
end 

w_vec = [w_1,w_2,w_3]; 

n = 600; 
% Genrates continuous uniform random numbers for X_train
X_train = unifrnd(-1,1,[n,d]); 

% Sets noise-parameters 
mu = 0;
sigma = 0.02;

[x_r,x_c] = size(X_train);  

% Generates response vector Yobs_train from fun_i and X_train
ytr_1 = fun_i(w_vec(:,1), X_train, sigma); 
ytr_2 = fun_i(w_vec(:,2), X_train, sigma); 
ytr_3 = fun_i(w_vec(:,3), X_train, sigma); 
Yobs_train = [ytr_1, ytr_2, ytr_3]; 

% Generates a new matrix with external data of dim 1000x2000 with continuos
% random numbers on hte interval -1,1
X_external = unifrnd(-1,1,[d/2,d]); 
[x_r_1000,x_c_1000] = size(X_external);

% Generates response vector Yobs_external from fun_i and X_external
Yobs_external = fun_i(w_vec, X_external, sigma); 


%% Task 2

% //////////////////////////////////////////////////////
% a) 

% Uses OLS (MLE) to predict the 2000 variables in w for each set generated
% in a) (ie: w1,w2,w3 for Yobs_train
% The results end up inaccurate or faulty as n < d for the data

%{
for i = 1:3
    mle_i = @(w_vec)ols_i(X_train, w_vec, Yobs_train(:,i));
    w_pred(:,i) = fminunc(mle_i, w_vec(:,i));
end
%}

% b)

% Sets a variable for the no. folds in the cross-validation
K = 9; 
% Generates 100 values for lambda on the interval 0,1 using logspace for a
% more realistic frequency distribution
lambda = logspace(-5,20,100); 

% Normalizes the values in X
zX = normalize(X_train);
% Normalizes the values in Y
zY = normalize(Yobs_train);
% Creates an inverted matrix
X_train_c_norm = zX';

% Create figure for sublots in loop.
figure()
% Generates vector for storing optimal values for lambda
lambda_opt = zeros(1,3);

for i=1:3
    % Calls the function fitrlinear which trains linear regression models using
    % predictor data by minimizing an objective function. 
    % The resulting CVMdl (Cross-Validated-Model) is a RegressionPartitionedLinear 
    % model which contains K RegressionLinear models trained on each fold in a
    % cell-vector, these are finally tested on the test-fold    
    CVMdl = fitrlinear(X_train_c_norm,zY(:,i),'ObservationsIn','columns','KFold',K,'Lambda',lambda,'Learner','leastsquares','Solver','sgd','Regularization','ridge');
    
    % The RMSE is then calculated through passing the CVMdl to kfoldloss to
    % calculate MSE and then sqrt for RMSE
    rmse = sqrt(kfoldLoss(CVMdl));
    % The min-values for lambda are then identified along with their
    % position using min
    [rmse_min,lambda_ind]=min(rmse);
    % Finally, these min-values are stored in the optimal vector earlier
    % created
    lambda_opt(i)=lambda(lambda_ind);
    
    % Plots the values of lambda against the values for RMSE.
    subplot(1,3,i)
    plot(log10(lambda),rmse)
    xlabel('log10(\lambda)')
    ylabel('RMSE')
    title({['Model ',num2str(i)]})
    % Add value of optimal lambda
    pos=get(gca,'Position');
    box_pos=[pos(1)+pos(3)-0.19,0.88];
    box_size=[0.01,0.01];
    annotation('textbox',[box_pos,box_size],'String',{['\lambda^*=',num2str(lambda_opt(i))]},'FitBoxToText','On');
    
end

%%

% Creates w_hat based on the size of w_vec
w_hat=zeros(size(w_vec,1),3);

% For-loop using the matlab function ridge to return a vector of coefficient
% estimates for a multilinear ridge regression of the responses in y on the
% predictors in X to store in w_hat. This is then scatterplotted against
% the no. dimensions
for i=1:3
    w_hat(:,i)=ridge(zY(:,i),zX,lambda_opt(i));
    figure()
    subplot(1,2,1)
    scatter(1:d,w_hat(:,i),'.');
    hold on
    scatter(1:d,w_vec(:,i),'.');
    title({['Ordered estimated (blue) and true'],[' (orange) weights of model ',num2str(i)]})
    xlabel('Index')
    ylabel('Value')
    
    subplot(1,2,2)
    % Randomly permutes the weights as to give a more realistic spread 
    % for the values
    perm_ind=randperm(length(w_vec(:,i)));
    scatter(1:d,w_hat(perm_ind,i),'.')
    hold on
    scatter(1:d,w_vec(perm_ind,i),'.');
    title({['Unordered estimated (blue) and true'],[' (orange) weights of model ',num2str(i)]})
    xlabel('Index')
    ylabel('Value')
end

% Takes the average of the predicted values for those with indices over 500, and
% those with indices below for model 1
w_les_500 = mean(w_hat(1:500,1));
w_gre_500 = mean(w_hat(501:1500,1));

disp(w_les_500);
disp(w_gre_500);

%% Task 3

%% a) 

% Which model do we want to work with (1-3)?
model_no = 1;
% Sets the no-folds in the inner and outer loop respectively
k_out = 9;
k_in = 8;

% re-define lambda
lambda = logspace(-2,2,100);
% Set the no. runs for the double cross-validation
j = length(lambda); 

% Creates empty vectors for the mean values of rmse
mean_rmse_in = zeros(1,k_out*j);
mean_rmse_out = zeros(1,k_out*j);
% Creates empty vectors for lambda and w
lambda_hat=zeros(1,k_out*j);
w_hat=zeros(d,k_out*j);

for i=1:j
    % Get permutation indeces.
    perm_ind = randperm(length(zY(:,model_no)));
    
    % Calls the double cross-validation function to obtain the desired
    % values
    [mean_rmse_in(1+(i-1)*k_out:i*k_out), mean_rmse_out(1+(i-1)*k_out:i*k_out),lambda_hat(1+(i-1)*k_out:i*k_out),w_hat(:,1+(i-1)*k_out:i*k_out)] = double_cv_fun(zY(perm_ind,model_no),zX(perm_ind,:),lambda,k_out,k_in);
end

%%
figure()
histogram(mean_rmse_in)
hold on
histogram(mean_rmse_out)
title({'Histograms of RMSE values from inner (blue) vs. outer (orange)',' cross-validation'});
xlabel('RMSE')
ylabel('# values')

pos=get(gca,'Position');
box_pos=[pos(1)+pos(3)-0.23,0.88];
box_size=[0.01,0.01];
annotation('textbox',[box_pos,box_size],...
    'String',{['\mu_{outer}=',num2str(mean(mean_rmse_out))],['\sigma^2_{outer}=',num2str(var(mean_rmse_out))]},...
    'FitBoxToText','On');

box_pos=[pos(1)+pos(3)-0.65,0.88];
box_size=[0.01,0.01];
annotation('textbox',[box_pos,box_size],...
    'String',{['\mu_{inner}=',num2str(mean(mean_rmse_in))],['\sigma^2_{inner}=',num2str(var(mean_rmse_in))]},...
    'FitBoxToText','On');
    

%% c) 

% Generates two large external test sets consisting of 4000 observations
% with respect to he fist model
X_big_ext = 1+2*rand(d*2,d);
y_big_ext = fun_i(w_1,X_big_ext,sigma);

% normalize
zX_big_ext = normalize(X_big_ext);
zY_big_ext = normalize(y_big_ext);

% for each weight vector from b), calculate RMSE on big external test set.
mse_big = zeros(1,j);
for i=1:j
    y_big_ext_hat = zX_big_ext*w_hat(:,i);
    % Calls the function immse to calculate the mean squared error
    mse_big(i) = immse(y_big_ext_hat,zY_big_ext);
end

% Calculates the RMSE through taking the square root of the value for MSE
rmse_big_ext = sqrt(mse_big);
figure()
histogram(rmse_big_ext,10)
hold on
histogram(mean_rmse_out,10)

title({'Histograms of RMSE values from external test set (blue) vs. outer (orange)',' test set from cross-validation'});
xlabel('RMSE')
ylabel('# values')

pos=get(gca,'Position');
box_pos=[pos(1)+pos(3)-0.23,0.88];
box_size=[0.01,0.01];
annotation('textbox',[box_pos,box_size],...
    'String',{['\mu_{outer}=',num2str(mean(mean_rmse_out))],['\sigma^2_{outer}=',num2str(var(mean_rmse_out))]},...
    'FitBoxToText','On');

box_pos=[pos(1)+0.005,0.88];
box_size=[0.01,0.01];
annotation('textbox',[box_pos,box_size],...
    'String',{['\mu_{external}=',num2str(mean(rmse_big_ext))],['\sigma^2_{external}=',num2str(var(rmse_big_ext))]},...
    'FitBoxToText','On');
%}
