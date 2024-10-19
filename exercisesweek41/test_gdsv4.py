import numpy as np
import matplotlib.pyplot as plt
import gradient_descent_ols_and_ridge as gd
import gradient_descent_ridge_version2 as gds

# Generate the dataset
np.random.seed(42)  # For reproducibility
n_samples, n_features = 100, 2
X = np.random.rand(n_samples, n_features)
true_beta = np.array([3, 5])
y = X @ true_beta + np.random.randn(n_samples) * 0.5  # Adding some noise

# Add a column of ones to X for the intercept term
X = np.c_[np.ones(n_samples), X]
true_beta = np.r_[1, true_beta]  # Adding the intercept term to true_beta

# Initial guess for beta
initial_beta = np.zeros(X.shape[1])

# Gradient Descent Parameters
learning_rate = 0.1
n_iterations = 1000

# Choose optimization algorithm
# approx could be 'adagrad', 'rmsprop', or 'adam'
epsilon = 1e-8
gamma = 0.9
momentum_gamma = 0.9
lmbda = 0.001

approx = gd.adam
momentum = True
model = gd.ridge
modelname = "ridge"
approx_name = "adam"

beta_gd = gd.plain_gradient_descent(X,y,initial_beta, learning_rate, n_iterations, model=model, approx=approx, momentum=momentum)
beta_gdS = gds.adam_momentum_gradient_descent_ridge(X, y, initial_beta, learning_rate, n_iterations, lmbda)


print(f"True beta: {true_beta}")
print(f"Beta using approximation function %s and model %s: {beta_gd}" %(str(approx_name), modelname))
print(f"Sophia's beta using approximation function %s and model %s: {beta_gdS}" %(str(approx_name), modelname))
print("Comparison my code and original %s:" %str(beta_gd==beta_gdS))