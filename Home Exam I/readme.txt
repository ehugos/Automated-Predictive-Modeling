run_assignment acts as the main script, calling all other functions automatically when run. 

poly_func represents the polynomial quadratic model, and is ran through the input parameters X 
(the concentration matrix), and the parameter vector p. It yields a vector of observed values for Y.

hill_func represents the Hill-based global response surface model, and is ran through the input parameters X 
(the concentration matrix), and the parameter vector par_v. It yields a vector of observed values for Y.

simp_hill_func represents the Simplified Hill-model, and is ran through the input parameters X 
(the concentration matrix), and the parameter vector par_v. It yields a vector of observed values for Y.

ols_poly_i calculates the ordinary least squares value for the observed vector y_obs, and the output from poly_func, given inputs 
X (concentration matrix), y_obs (vector for observed y-values and q_p (parameter vector).

ols_hill_i calculates the ordinary least squares value for the observed vector y_obs, and the output from hill_func, given inputs 
X (concentration matrix), y_obs (vector for observed y-values and x_p (parameter vector).

ols_simp_hill_i calculates the ordinary least squares value for the observed vector y_obs, and the output from simp_hill_func, given inputs 
X (concentration matrix), y_obs (vector for observed y-values and w_p (parameter vector).

r_squared_func calculates the coefficient of determination given inputs obs (observed  values) and pred (predicted values)