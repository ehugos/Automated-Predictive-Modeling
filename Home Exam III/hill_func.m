function [y_obs] = hill_func( X, par_v )
% Function for the combined hill-function polynomial

[X_r,X_c] = size(X);

% Creates the proportion variable vectors theta
for i= 1:X_r
    theta_v(i,:) = X(i,:)./ sum(X(i,:)); 
end

% Creates gamma and ic50 as a function of theta
% Creates c, the concentration vector in the hill-model based upon the sum
% of its rows
% Performs calculations for the modified hill-model based upon these
% constants
ic_50 = par_v(1) + par_v(2)*theta_v(:,1) + par_v(3)*theta_v(:,2) + (par_v(4)*theta_v(:,1).^2) + (par_v(5)*theta_v(:,2).^2) + (par_v(6)*theta_v(:,1).*theta_v(:,2));
gamma = par_v(7) + par_v(8)*theta_v(:,1) + par_v(9)*theta_v(:,2) + (par_v(10)*theta_v(:,1).^2) + (par_v(11)*theta_v(:,2).^2) + (par_v(12)*theta_v(:,1).*theta_v(:,2));

for i = 1:X_r  
    y_obs(i) = 1/(1 +((sum(X(i,:))/ic_50(i))^gamma(i)));
end

% Replaces any NAN numbers with 1
y_obs(isnan(y_obs))= 1;
y_obs = y_obs';
end

