function [ q ] = ols_poly_i( q_p, X, y_obs )
% Defines the OLS function for the quadratic model

[x_r, x_c] = size(X);
q = sum(((y_obs - poly_func(X,q_p)').^2)/x_r);

end

