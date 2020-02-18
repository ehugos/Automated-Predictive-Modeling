function [ x ] = ols_hill_ii( x_p, X, y_obs )
% Defines the OLS function for the quadratic model

[x_r, x_c] = size(X);
x = sum((y_obs - hill_func(X,x_p)').^2)/x_r;

end
