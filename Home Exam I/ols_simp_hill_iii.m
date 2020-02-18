function [ w ] = ols_simp_hill_iii( w_p, X, y_obs )
% Defines the OLS function for the quadratic model

[x_r, x_c] = size(X);
w = sum( ((y_obs - simp_hill_func(X,w_p)').^2)/x_r );

end
