function [ x_discrp, y_discrp, std_net ] = discr_fun( sexp, y_obs, net_cell, opt_hid)
   
    % Uses the data obtained to create a set of optimal models for
    % the dataset based on the optimal value for no. hidden layers
    net_1 = net_cell{opt_hid,1};
    net_2 = net_cell{opt_hid,2};
    net_3 = net_cell{opt_hid,3};
    net_4 = net_cell{opt_hid,4};
    net_5 = net_cell{opt_hid,5};
    net_6 = net_cell{opt_hid,6};
    net_7 = net_cell{opt_hid,7};
    net_8 = net_cell{opt_hid,8};
    net_9 = net_cell{opt_hid,9};
    net_10 = net_cell{opt_hid,10};
    
    %%
    % v)
    
    % Runs the 512 variable input through the trained optimal models  
    y_net_1 = net_1(sexp');
    y_net_2 = net_2(sexp');
    y_net_3 = net_3(sexp');
    y_net_4 = net_4(sexp');
    y_net_5 = net_5(sexp');
    y_net_6 = net_6(sexp');
    y_net_7 = net_7(sexp');
    y_net_8 = net_8(sexp');
    y_net_9 = net_9(sexp');
    y_net_10 = net_10(sexp');
    
    % Stores all 512x10 response values in a new matrix
    y_net_m = [y_net_1 ; y_net_2; y_net_3; y_net_4; y_net_5; y_net_6; y_net_7; y_net_8; y_net_9; y_net_10];
    
    
     % Calculates the standard deviation for each row (example)
    [ynr, ync]= size(y_net_m);
    
    % Calculates the standard deviation for each row(example)
    for m = 1:ync
        std_net(m,1) = std(y_net_m(:,m));
    end
    
    % Selects the 30 values with the largest discrepancy, and also obtains
    % their index position (ie: example no.)
    % [max_discrp_2, max_discrp_ind_2] = maxk(std_net, 30) not working on
    % R2017a   
    [discrpval, discrpind] = sort(std_net, 'descend');
    max_discrp_ind = discrpind(1:30);
    x_discrp = sexp(max_discrp_ind,:);
    y_discrp = y_obs(max_discrp_ind,:);
end

