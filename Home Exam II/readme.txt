run_assignment is the main script for Home-exam 2.
It firs generates data, folowed by Ridge-regression parameter estimation for model 1,2 & 3.
It then plots the estimated parameters against the true values. This is followed by double cross validation for 
ridge regression parameter estimation for model 1 (outer folds = 9, inner folds = 8) performed 100 times. This is 
then compared to´a new large generated dataset, both models have their RMSE plotted as histograms. This script will call 
fun_i and double_cv_fun

fun_i is a function generation the response vector y for inputs w (a 2000-dimensional parameter vector), x_n (input matrix), sigma on the form yi(n) = wiT*xn + err where err is a 
random number vector of the same dimesion as y, from the normal distribution with mean parameter 0 and standard deviation parameter sigma.

double_cv_fun is a function for performing double cross validation on the given in-parameters y,X,lambda,k_out,k_in. Where y is the output, x is the input, 
lambda is the penalty parameter vector for ridge-regression & k_in, k_out are the values for the inner & outer fold in the double cross validation.