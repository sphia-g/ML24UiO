import numpy as np
import matplotlib.pyplot as plt
from gradient_descent_version3 import plain_gradient_descent, adagrad, rmsprop, adam
from gradient_descent_version2 import plain_gradient_descent as pg, plain_gradient_descent_adagrad, momentum_gradient_descent_adagrad, adam_plain_gradient_descent

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

# Run the gradient descent
beta_gd = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations)
beta_adagrad = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations, approx=adagrad)
beta_rmsprop = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations, approx=rmsprop)
#beta_adam = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations, approx=adam)
beta_adam = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations, gamma, momentum_gamma=0.999)
## beta1=0.9, beta2=0.999
beta_adagrad_momentum = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations, momentum=True,approx=adagrad)
beta_rmsprop_momentum = plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations, momentum=True, approx=rmsprop)
beta_pg_S = pg(X, y, initial_beta, learning_rate, n_iterations)
beta_adagrad_S = plain_gradient_descent_adagrad(X, y, initial_beta, learning_rate, n_iterations, adagrad=True)
beta_adagrad_momentum_S = momentum_gradient_descent_adagrad(X, y, initial_beta, learning_rate, n_iterations, gamma, adagrad=True)
beta_adam_S = adam_plain_gradient_descent(X, y, initial_beta, learning_rate, n_iterations)

# Print the results
print(f"True beta: {true_beta}")
print(f"Beta (Gradient Descent): {beta_gd}")
print(f"Sophia's Beta (Gradient descent): {beta_pg_S}" )
print(f"Beta (AdaGrad): {beta_adagrad}")
print(f"Beta (AdaGrad with momentum): {beta_adagrad_momentum}")
print(f"Sophia's Beta (AdaGrad): {beta_adagrad_S}")
print(f"Sophia's Beta (AdaGrad with momentum): {beta_adagrad_momentum_S}")
print(f"Beta (RMSProp): {beta_rmsprop}")
print(f"Beta (RMSprop with momentum): {beta_rmsprop_momentum}")
print(f"Beta (Adam): {beta_adam}")
print(f"Sophia's beta (Adam): {beta_adam_S}")


# Optionally, visualize the results:
plt.scatter(X[:, 1], y, alpha=0.5, label='Data')

x_line = np.linspace(0, 1, 100)
"""
# Equation: y = intercept + slope * x
plt.plot(x_line, 1 + 3 * x_line, 'g--', label='True Line')  # True beta
plt.plot(x_line, beta_gd[0] + beta_gd[1] * x_line, 'r-', label='GD Line')
plt.plot(x_line, beta_adagrad[0] + beta_adagrad[1] * x_line, 'b-', label='AdaGrad Line')
plt.plot(x_line, beta_rmsprop[0] + beta_rmsprop[1] * x_line, 'c-', label='RMSProp Line')
plt.plot(x_line, beta_adam[0] + beta_adam[1] * x_line, 'm-', label='Adam Line')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
"""

##plt.plot(x_line, beta_adagrad[0] + beta_adagrad[1] * x_line, 'b-', label='AdaGrad Line')
plt.plot(x_line, beta_adagrad_momentum[0] + beta_adagrad_momentum[1] * x_line, '-', label='AdaGrad Line momentum')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()