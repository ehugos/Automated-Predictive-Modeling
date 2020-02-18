function [res] = poly_func(X, p )
% Function for the combined hill-function polynomial

% Assigns the theta and gamma values based on the parameter vector
% according to the polynomial model

c1 = X(:,1);
c2 = X(:,2);
c3 = X(:,3);

% Loops for the indexes of i and ij based upon the no. of concentration
% vectors in X

res = p(1) + (p(2)*c1) + (p(3)*c2) + (p(4)*c3) + (p(5)*(c1.^2)) + (p(6)*(c2.^2)) + (p(7)*(c3.^2)) + (p(8)*c1.*c2) + (p(9)*c1.*c3) + (p(10)*c2.*c3);

end