clear all;
close all;

% By Hugo Swenson (husw7546), Uppsala University HT19

%%

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

%%

% Task 3

%%

% ii)

% Performs k-means clustering on the dataset
k_exp = 30;
sexp = X;
s_train = [];

[idx_train, centroid_l] = kmeans(sexp,k_exp);
% Chooses one example from each cluster and adds to new vector
for i = 1:k_exp
    % Selects all points with value as i
    data_p = sexp(idx_train == i,:);
    y_p = y_obs(idx_train == i,:);
    % Randomly selects data values from the size
    data_pos = randperm(size(data_p,1));
    % Retrieves values from the sets
    data_p = data_p(data_pos,:);
    y_p = y_p(data_pos,:);
    % Adds the new values to the dataset
    s_train_i = [data_p(1,:) y_p(1,:)];
    s_train = [s_train ; s_train_i];
end

%%

% iii)

for i = 1:5
    % iv)
    
    % Calls the net_train function to train the dataset using
    % cross-validation
    [ net_cell, opt_ind, opt_val, rmse ] = net_train(s_train);
    % Stores the optimal network index each run
    opt_mod_lay(i) = opt_ind;
    % Calls the qbc_fun function as to obtain the max discrepancy for x,y
    % as well as the standard deviation for the network
    [ x_discrp, y_discrp, std_net ] = discr_fun(sexp, y_obs, net_cell, opt_mod_lay(i));
    
    %%
    % vi)
    
    % Performs a check on the 30 values with the largest discrepancy, if this
    % statement is false, these values are added onto s_train, and iii) is
    % repeated
    if max(std_net) < 0.05
        disp('Standard deviation for max below treshhold, breaking');
        break;
    elseif  i == 5
        disp('5 runs completed, breaking');
        break;
    else
        fprintf('Run no %d has finished, adding 30 experiments', i);
        s_train = [s_train; [x_discrp y_discrp]];
    end
    
end

%%

% Task 4

% Generates new noisy response variables
err_i = normrnd(mu,sigma,y_r,y_c);
y_obs = y + err_i;

% i)

% Generates a dataset with 30 examples
s_it = [];
for i = 1:k_exp
    % Selects all points with value as i
    data_p_150 = sexp(idx_train == i,:);
    y_p_150 = y_obs(idx_train == i,:);
    % Randomly selects data values from the size
    data_pos_150 = randperm(size(data_p_150,1));
    % Retrieves values from the sets
    data_p_150 = data_p_150(data_pos_150,:);
    y_p_150 = y_p_150(data_pos_150,:);
    % Adds the new values to the dataset
    s_train_i_150 = [data_p_150(1,:) y_p_150(1,:)];
    s_it = [s_it ; s_train_i_150];
end

% Uses QCP to generate 150 training examples, starts with 30 selected
% through k-means clustering, then adds 30 new to the dataset based on
% maximum discrepancy.
for i = 1:4
    % Calls the net_train function to train the dataset using
    % cross-validation
    [ net_ind_150, opt_ind_150 ] = net_train(s_it);
    % Stores the optimal network index each run
    opt_hid_150(i) = opt_ind_150;
    % Calls the qbc_fun function as to obtain the max discrepancy for x,y
    % as well as the standard deviation for the network
    [ x_discrp_150, y_discrp_150, std_net_150 ] = discr_fun( sexp, y_obs, net_ind_150, opt_hid_150(i));
    fprintf('Run no %d has finished, adding 30 experiments', i);
    s_it = [s_it; [x_discrp_150 y_discrp_150]];
end
[ net_cell_it, opt_ind_it, opt_val_it, rmse_it ] = net_train(s_it);
min_it = min(min(rmse_it));
[ind_x_it, ind_y_it] = find(rmse_it == min_it);
bestnet_it = net_cell_it{ind_x_it, ind_y_it};

% ii)

% Selects 150 random pairs from the dataset
[s_rand_c, rand_idx] = datasample(sexp(:,1),150);
s_rand = zeros(150,4);
s_rand(:,1) = s_rand_c';
s_rand(:,2) = sexp(rand_idx,2);
s_rand(:,3) = sexp(rand_idx,3);
s_rand(:,4) = y_obs(rand_idx);
[ net_cell_rand, opt_ind_rand, opt_val_rand, rmse_rand ] = net_train(s_rand);
min_rand = min(min(rmse_rand));
[ind_x_rand, ind_y_rand] = find(rmse_rand == min_rand);
bestnet_rand= net_cell_rand{ind_x_rand, ind_y_rand};
% iii

% Uses D-optimal design to select 150 rows from sexp, then takes these rows
% and generates a dataset
r_list = candexch(sexp, 150);
% The D-optimal design selects the same three datasets, in a repeating
% sequence. I am not sure how to resolve this error, hence this yields a
% poor model.
s_d_opt = [sexp(r_list,1),sexp(r_list,2), sexp(r_list,3), y_obs(r_list)];
[ net_cell_d, opt_ind_d, opt_val_d, rmse_d ] = net_train(s_d_opt);
min_d = min(min(rmse_d));
[ind_x_d, ind_y_d] = find(rmse_d == min_d);
bestnet_d = net_cell_d{ind_x_d, ind_y_d};

% Creates 3 linearely spaced vectors with 100 datapoints each
lc_1 = linspace(0,300, 100);
lc_2 = linspace(0,100, 100);
lc_3 = linspace(0,300, 100);
% Creates a ndgrid with 10 000 datapoints
[sexp_large_1,sexp_large_2,sexp_large_3] = ndgrid(lc_1,lc_2,lc_3);
sexp_large = [sexp_large_1(:) , sexp_large_2(:) , sexp_large_3(:)];

y_obs_large = hill_func(sexp_large, n_m);

% Uses the nets to generate 10000 predicted values for each net
% respectively
net_val_it = bestnet_it(sexp_large');
net_val_rand = bestnet_rand(sexp_large');
net_val_d = bestnet_d(sexp_large');

% Calculates the RMSE for each network with respect to the large dataset
rmse_it = sqrt(perform(bestnet_it, y_obs_large, net_val_it'));
rmse_rand = sqrt(perform(bestnet_rand, y_obs_large, net_val_rand'));
% Does not take into account the values for the D-optimal design as its
% Model selection is skewed
% rmse_d_p = sqrt(perform(bestnet_d, y_obs_large, net_val_d'));

% Displays the model with the smallest RMSE value
best_mod = min(rmse_it, rmse_rand);
if best_mod == rmse_it
    disp('The iterative k-means method yields the best network-model')
elseif best_mod == rmse_rand
    disp('The random-selection method yields the best network-model')
else
    error('Something went terribly wrong')
end

%%

% Task 5




