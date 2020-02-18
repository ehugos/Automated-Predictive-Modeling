clear all; 
close all; 

% By Hugo Swenson (husw7546), Uppsala University HT19

% ////////////////////////////////////////////////////////////
% Task 2

% a 

% Assigns the concentration vectors for the drugs for the surface plot
c1_2_par = linspace(0,300);
c2_2_par = linspace(0,100);

% Creates a concentration matrix for all possible concentration combinations 
[caf,cbf] = ndgrid(c1_2_par,c2_2_par);
X_2_par = [caf(:), cbf(:)];

% Sets the estimated parameters for cancerous and non-cancerous cells
% n stands for "normal ("healthy").
n_m = [117.11, -15.71, -86.15 , 79.55, 42.67, -33.75, 1.70, -1.02, 0.41 , 0.31, -0.74, 1.21];

% Calculates the value for Y with gamma and ic50 as inputs
y_obs_2_par = hill_func(X_2_par, n_m);

% Reshapes the 1x64 matrix into a 8x8 dimensional matrix
y_obs_2_par = reshape(y_obs_2_par,[100,100]);

% ////////////////////////////////////////////////////////////
% b

% Creates a 3d surface plot for the response
figure('Name', 'Response_Surface_2_Para');
xlabel('AG490 concentration [µM]')
ylabel('U0126 concentration [µM]')
surf(c1_2_par,c2_2_par ,y_obs_2_par);

% ////////////////////////////////////////////////////////////
% c

% Performs the same operations as in a, but with all variables present
c1 = [0, 0.3, 1, 3, 10, 30, 100, 300];
c2 = [0, 0.1, 0.3, 1, 3, 10, 30, 100];
c3 = [0, 0.3, 1, 3, 10, 30, 100, 300];

[cac,cbc,ccc] = ndgrid(c1,c2,c3);
X = [cac(:), cbc(:),ccc(:)]; 

% Calculates the value for y_observed with the full 512x3 combinations of
% concentrations
y = hill_func(X, n_m);
[y_r, y_c] = size(y);

% Generates the error term from 0 means with deviation sigma (corresponding
% to 2 % of the max value of the response variabel (ic50) from a normal distribution
% and adds this to the term y_observed
mu=0;
sigma = 0.02;
err = normrnd(mu,sigma,y_r,y_c);
y_obs = y + err;

% ////////////////////////////////////////////////////////////
% Task 3 

% a)

% Store the cocnentration from X into 10 separate variables based on the
% structure of the Beta parameter from the quadratic polynomial model
for i=1:length(X)
beta_raw(:,i) = [1; X(i,1); X(i,2); X(i,3); X(i,1).*X(i,2); X(i,1).*X(i,3); X(i,2).*X(i,3); X(i,1).^2; X(i,2).^2; X(i,3).^2]; 
end

% Calculate the parameter values using Linear algebra for model estimatuon via the formula: Beta = (X'*X)X'Y
beta_v = inv(beta_raw*beta_raw')*beta_raw*y_obs;
% Sets the starting value to 10% from the true value
q0 = 0.9*beta_v;

% Calculates new estimated parameters by first calling the OLS function and then using fminsearch to estimate new parameters 
par_fun_i = @(q_p)ols_poly_i(q_p, X, y_obs');
% Tries to minimize the function given the input, restricted by the newly
% assigned options variable
par_i = fminunc(par_fun_i, q0);

% Uses the new estimated parameters to calculate the hill-function
Y_poly_est = poly_func(X, par_i);
% Calculates the R^2 value compared to the original y_observations
R2_poly = r_squared_func(y_obs, Y_poly_est);

% ////////////////////////////////////////////////////////////
% b) 

% Sets the value for the initial guess to a 10% difference from the true
% value of the parameter vector
x0 = 0.9*n_m;

% Defines the OLS function
par_fun_ii = @(q_p)ols_hill_ii(q_p, X, y_obs');
% Tries to minimize the function given the input, restricted by the newly
% assigned options variable
par_ii = fminunc(par_fun_ii, x0);

% Uses the new estimated parameters to calculate the hill-function
Y_hill_est = hill_func(X, par_ii);
% Calculates the R^2 value compared to the original y_observations
R2_hill = r_squared_func(y_obs, Y_hill_est);

% ////////////////////////////////////////////////////////////
% c)

% Uses the new estimated parameters to calculate the simplified hill-function

% Assigns the parameters alfa and w, w = 0,0,0 since the initial drug
% response model starts in origo
alfa = 1;
w0 = [0,0,0,0,alfa];

% Calculates new estimated parameters by first calling the OLS function and then using fminsearch to estimate new parameters 
par_fun_iii = @(w_p)ols_simp_hill_iii(w_p, X, y_obs');
% Tries to minimize the function given the input, restricted by the newly
% assigned options variable
par_iii = fminunc(par_fun_iii, w0);

% Uses the new estimated parameters to calculate the hill-function
Y_simp_hill_est = simp_hill_func(X, par_iii);

% Calculates the R^2 value compared to the original y_observations
R2_simp_hill = r_squared_func(y_obs, Y_simp_hill_est);

% ////////////////////////////////////////////////////////////
% Task 4

% a) 

figure('Name', 'Scatterplot_Comparisons');
subplot(1,3,1);
    scatter(y_obs,Y_poly_est,'.')
    title('Polynomial-Model')
    xlabel('Observed ATP levels')
    ylabel('Estimated ATP levels')
    legend(strcat('R^2: ', num2str(R2_poly)))
subplot(1,3,2);
    scatter(y_obs,Y_hill_est,'.','r')
    title('Hill-Model')
    xlabel('Observed ATP levels')
    ylabel('Estimated ATP levels')
    legend(strcat('R^2: ', num2str(R2_hill)))

subplot(1,3,3);
    scatter(y_obs,Y_simp_hill_est,'.', 'm')
    title('Simplified Hill-model')
    xlabel('Observed ATP levels')
    ylabel('Estimated ATP levels')
    legend(strcat('R^2: ', num2str(R2_simp_hill)))

% ////////////////////////////////////////////////////////////    
% b) 

%  Generates a new dataset of 512 examples
Y = hill_func(X, n_m);
[y_r_i, y_c_i] = size(Y);
% Generates the error term from 0 means with deviation sigma (corresponding
% to 2 % of the max value of the response variabel (ic50) from a normal distribution
err_i = normrnd(mu,sigma,y_r_i,y_c_i);
Y_obs = Y + err_i;

% Calculates new R2 values for the external data comparisons
R2_Poly_i = r_squared_func(Y_obs, Y_poly_est);
R2_Hill_i = r_squared_func(Y_obs, Y_hill_est);
R2_Simp_i = r_squared_func(Y_obs, Y_simp_hill_est);

figure('Name', 'External_Data_Comparisons');
subplot(1,3,1);
    scatter(Y_obs,Y_poly_est,'.')
    title('Polynomial-Model')
    xlabel('Observed ATP levels')
    ylabel('Estimated ATP levels')
    legend(strcat('R^2: ', num2str(R2_Poly_i)))
subplot(1,3,2);
    scatter(Y_obs,Y_hill_est,'.','r')
    title('Hill-Model')
    xlabel('Observed ATP levels')
    ylabel('Estimated ATP levels')
    legend(strcat('R^2: ', num2str(R2_Hill_i)))

subplot(1,3,3);
    scatter(Y_obs,Y_simp_hill_est,'.', 'm')
    title('Simplified Hill-model')
    xlabel('Observed ATP levels')
    ylabel('Estimated ATP levels')
    legend(strcat('R^2: ', num2str(R2_Simp_i)))
    
% ////////////////////////////////////////////////////////////
% Task 5

% a

%Calculates the relative error between the true parameter vector n_m and the
%estimated parameter vector para_ii
rel_err = abs(n_m - par_ii)/abs(n_m);
disp(rel_err); 

% ////////////////////////////////////////////////////////////
% b 

% Loops and creates 99 new concentration matrixes to add to the existing
for i = 1:99
   Y_obs(:,i+1) = hill_func(X, n_m)'; 
   err_ii = normrnd(mu,sigma,512,1);
   Y_obs(:,i+1) = Y_obs(:,i+1) + err_ii;
end

for i = 1:100
% Calculates new estimated parameters by first calling the OLS function and then using fminsearch to estimate new parameters 
fun_par_100 = @(par_v)ols_hill_ii(par_v, X, Y_obs(:,i)');
% Tries to minimize the function given the input, restricted by the newly
% assigned options variable
Theta_hat(:,i) = fminunc(fun_par_100, x0);
end

% ////////////////////////////////////////////////////////////
% c
% Calculates the relative error of Theta hat for each parameter

for i=1:12
% diffs(i,:) = Theta_hat(i,:) - par_ii(i);  
rel_Theta_err(i,:) = norm(Theta_hat(i,:) - par_ii')/norm(Theta_hat(i,:));
    for j=1:100
        rel_Theta_err(i,j) = norm(Theta_hat(i,j) - par_ii(i))/norm(Theta_hat(i,j)); 
    end
% Calculates the interquartile-range 
iqr_Theta(i)=iqr(rel_Theta_err(i,:)); 
end

% Plots the iqr for Theta_hat
figure('Name','iqr for Theta_hat')
plot(iqr_Theta)
xlabel('Parameters'); 
ylabel('IQR'); 
title('IQR of the relative error');

% Plots a box-plot for the relative error of each parameter in Theta    
figure('Name','Relative errors for the b-parameters')
for i = 1:6
     subplot(2,3,i)
     boxplot(rel_Theta_err(i,:));
     xlabel('parameter b ')
end

figure('Name','Relative errors for the a-parameters');
for i = 1:6
     subplot(2,3,i)
     boxplot(rel_Theta_err(i+6,:));
     xlabel('parameter a ')
end
    
% ////////////////////////////////////////////////////////////
% Task 6

Y_sum = [];
X_sum = [];
no_runs = 3;
% Generates a new dataset of 512 examples, measured x number of times
% Adds noise on to this dataset

for i =1:no_runs
    
    err_iii = normrnd(mu,sigma,length(y_obs),1);
    Y_sum = [Y_sum ; (y_obs + err_iii)];
    X_sum = [X_sum ; X];
    
    par_fun_ii_inc = @(p_v)ols_hill_ii(p_v, X_sum, Y_sum');
    % Tries to minimize the function given the input, restricted by the newly
    % assigned options variable
    inc_par = fminunc(par_fun_ii_inc, x0);
    rel_err_inc(i) = abs(n_m - inc_par)/abs(n_m);
end

% Creates a X-axis for the plot with 512, 1024 and 1536 experiments 
no_runs_X = [];
for i=1:no_runs
    no_runs_X = [no_runs_X 512*i];
end

figure('Name', 'Relative error for 512 combinations')
    bar(no_runs_X, rel_err_inc);
    xlabel('# Experiments');
    ylabel('Relative error');
    
% ////////////////////////////////////////////////////////////
% Task 7

% Creates concentration vectors for three concentrations
c1_few = [0 ,3 ,300]; 
c2_few = [0 ,1 ,100]; 
c3_few = [0 ,3 ,300]; 

% Creates the concentration matrix (3^3 = 27 combinations)
[caf, cbf, ccf] = ndgrid(c1_few, c2_few, c3_few);
X_few = [caf(:), cbf(:), ccf(:)]; 

% Runs the hill function to obtain y_observed for the new matrix
Y_few = hill_func(X_few, n_m);
rel_err_few = normrnd(mu,sigma,length(Y_few),1);
Y_few = Y_few + rel_err_few;

Y_few_sum = [];
X_few_sum = [];

for i =1:no_runs
    
    err_iv = normrnd(mu,sigma,length(Y_few),1);
    Y_few_sum = [Y_few_sum ; (Y_few + err_iv)];
    X_few_sum = [X_few_sum ; X_few];
    
    par_fun_ii_dec = @(p_v)ols_hill_ii(p_v, X_few_sum, Y_few_sum');
    % Tries to minimize the function given the input, restricted by the newly
    % assigned options variable
    dec_par = fminunc(par_fun_ii_dec, x0);
    rel_err_dec(i) = abs(n_m - dec_par)/abs(n_m);
end

% Creates a X-axis for the plot with 27, 54 and 81 experiments 
no_runs_X_dec = [];
for i=1:no_runs
    no_runs_X_dec = [no_runs_X_dec 27*i];
end

figure('Name', 'Relative error for 27 combinations')
    bar(no_runs_X_dec, rel_err_dec);
    xlabel('# Experiments');
    ylabel('Relative error');