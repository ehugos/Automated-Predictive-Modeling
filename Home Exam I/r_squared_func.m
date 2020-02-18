function [ R2 ] = r_squared_func(obs ,pred )
% Calculates the R2 value for the obs and pred datasets

R2 = 1 - (sum((obs-pred).^2) / sum((obs-mean(obs)).^2));

end

