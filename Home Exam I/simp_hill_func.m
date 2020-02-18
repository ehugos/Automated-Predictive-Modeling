function [ res ] = simp_hill_func( X, w )
% Function for the simplified hill-function 

[X_r,X_c] = size(X);

for i = 1:X_r    
    res(i) = 1/((1 + w(1) + w(2)*X(i,1) + w(3)*X(i,2) + w(4)*X(i,3))^w(5));
end
res = res';
end

