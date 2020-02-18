function [ net_cell, opt_ind, opt_val, rmse ] = net_train(s_train)
[s_r, s_c] = size(s_train);

% No. folds
K = 10;
% Generates the vector v with values H = 1,2,3,4,5,10 (no. hidden layers in network)
v = [1,2,3,4,5,10];

% Partitions the dataset into 10 folds (9 training, one test, size per fold
% is equal to 3 (30/10 = 3)
cv = cvpartition(s_r,'KFold',K);
% Creates test and training data for the model to use with column 1-3 being
% the inputs (X) and column 4 being theobserved values for S_star_train
for j = 1:K
    trainIdxs{j} = find(training(cv,j));
    testIdxs{j} = find(test(cv,j));
    train_mat{j} = [s_train(trainIdxs{j},1:3) s_train(trainIdxs{j},4)];
    test_mat{j} = [s_train(testIdxs{j},1:3) s_train(testIdxs{j},4)];
end

% Trains each model set using training data, then evaluates. Retains the
% optimal value for H for each model

for k = 1:length(v)
    % Performs a 10-fold cross validation for hidden-layer k
    for l = 1:K
        % Calls the net for the different values on hidden layers
        net = feedforwardnet(v(k));
        net.layers{2}.transferFcn ='logsig';
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;
        % Trains the dataset using cross-validation
        [net,tr] = train(net, train_mat{l}(:,1:3)', train_mat{l}(:,4)');
        % Calculates the RMSE for each iteration (each cross validation
        % for H = 1,2,3,4,5), perform yields MSE as output
        y_p = net(test_mat{l}(:,1:3)');
        % Calculates the rmse for the net
        rmse(k,l) = sqrt(perform(net, test_mat{l}(:,4)', y_p));
        % Stores the model in the cell net_ind (10 for each value for hidden layer)
        net_cell{k,l} = net;
    end
    % Calculates the average rmse for the 10 networks generates with hidden
    % layers = k
    rmse_val(k) = mean(rmse(k,:));
end

% Obtains the smallest rmse and its index value, ie: that of the most optimal
% value for hidden layers
[opt_val, opt_ind] =  min(rmse_val);
end

