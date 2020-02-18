function [rmse_mean_in,rmse_mean_out,lambda_hat,w_hat] = double_cv_fun(y,X,lambda,k_out,k_in)

% Function that performce double cross validation for fitting regression line
% using to ridge as regularization with lambda as penalty parameter, on
% data, where y is observation of X, and X holds the samples in rows and
% features as columns. X and y are assumed to be normalized. k1 and k2 are
% the number of loops used in the inner and outer cv, respectively.

% Determine number of samples in y and X and dimension of X.
[x_r,x_c]=size(X);

% Create empty vectors/matrices for data soon to be appended
rmse_mean_in=zeros(1,k_out);
mse_out=zeros(1,k_out);
lambda_hat=zeros(1,k_out);
w_hat=zeros(x_c,k_out);

% Create cv-partitions for the inner and outer loop based on the sizes of
% the data-sets
k_part = cvpartition(x_r,'kFold',k_out);

for i=1:k_out
    X_out=X(test(k_part,i),:);
    y_out=y(test(k_part,i));
    X_in=X(training(k_part,i),:);
    y_in=X(training(k_part,i));
    
    % Inner k2-fold cross validation
    % Calls the function fitrlinear which trains linear regression models using
    % predictor data by minimizing an objective function. 
    % The resulting CVMdl (Cross-Validated-Model) is a RegressionPartitionedLinear 
    % model which contains K RegressionLinear models trained on each fold in a
    % cell-vector, these are finally tested on the test-fold
    % Transpose of X for faster performance (512 vs. 3).
    X_in_c = X_in';
    CVMdl_in = fitrlinear(X_in_c,y_in,'ObservationsIn','columns','KFold',k_in,'Lambda',lambda,'Learner','leastsquares','Solver','sgd','Regularization','ridge');
    % Calculate inner RMSE by taking the sqrt of the MSE obtained by
    % function KfoldLoss
    rmse_in = sqrt(kfoldLoss(CVMdl_in));
    % Obtains the smallest value for RMSE and then stores lambda based upon its
    % index no. 
    [rmse_mean_in(i),ind]=min(rmse_in);
    lambda_hat(i)=lambda(ind);
    % Utlizes the obtained bets value for lambda to estimate w using ridge
    % regression
    w_hat(:,i)=ridge(y_in,X_in,lambda_hat(i));
    % The obtained w is then used for prediction, the results stored in
    % y_hat_out
    y_hat_out=X_out*w_hat(:,i);
    % These values are then used to calculate the outer mse
    mse_out=immse(y_hat_out,y_out);
end

% Calculates the outer RMSE by taking the sqrt of the mse_out values
% previously obtained
rmse_mean_out = sqrt(mse_out);

