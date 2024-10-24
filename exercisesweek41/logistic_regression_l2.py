def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def compute_logistic_cost_l2(X, y, beta, lmbda):
    """
    Compute the binary cross-entropy cost with L2 regularization.

    Parameters:
    X : numpy array of shape (m, n), the design matrix
    y : numpy array of shape (m,), true labels (0 or 1)
    beta : numpy array of shape (n,), the model weights including intercept
    lmbda : scalar, the regularization strength
    
    Returns:
    cost : scalar, the cost with L2 regularization
    """
    m = X.shape[0]
    h = sigmoid(np.dot(X, beta))  # Predictions using sigmoid
    
    # Binary cross-entropy cost
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    # Add L2 regularization (excluding bias term beta[0])
    l2_regularization = (lmbda / (2 * m)) * np.sum(beta[1:] ** 2)
    
    # Total cost
    total_cost = cost + l2_regularization
    
    return total_cost
    
def compute_logistic_gradient_l2(X, y, beta, lmbda):
    """
    Compute the gradient for logistic regression with L2 regularization.
    
    Parameters:
    X : numpy array of shape (m, n), the design matrix
    y : numpy array of shape (m,), true labels (0 or 1)
    beta : numpy array of shape (n,), the model weights including intercept
    lmbda : scalar, the regularization strength
    
    Returns:
    gradient : numpy array of shape (n,), the gradient with respect to beta
    """
    m = X.shape[0]
    h = sigmoid(np.dot(X, beta))  # Predictions using sigmoid
    
    # Gradient of binary cross-entropy
    gradient = (1 / m) * np.dot(X.T, (h - y))
    
    # Add L2 regularization (excluding bias term beta[0])
    gradient[1:] += (lmbda / m) * beta[1:]
    
    return gradient

def stochastic_gradient_descent_logistic_l2(X, y, beta, learning_rate, n_iterations, lmbda, approx=None, momentum=False):
    """
    Stochastic Gradient Descent for Logistic Regression with L2 regularization.

    Parameters:
    X : numpy array of shape (m, n), the design matrix
    y : numpy array of shape (m,), true labels (0 or 1)
    beta : numpy array of shape (n,), initial weights (including intercept)
    learning_rate : scalar, learning rate for SGD
    n_iterations : int, number of iterations
    lmbda : scalar, the L2 regularization strength
    approx : function, approximation method (optional)
    momentum : bool, whether to use momentum (optional)
    
    Returns:
    beta : numpy array of shape (n,), the learned weights
    """
    m = X.shape[0]  # Number of training examples
    
    for iteration in range(n_iterations):
        # Shuffle the dataset (optional but improves convergence)
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(m):
            xi = X_shuffled[i, :].reshape(1, -1)  # Current example (1, n)
            yi = y_shuffled[i]  # True label (0 or 1)
            
            # Compute the gradient using the current example (with L2 regularization)
            gradient = compute_logistic_gradient_l2(xi, yi, beta, lmbda)
            
            # Update weights (standard SGD)
            beta -= learning_rate * gradient
            
            # You can include an approximation method or momentum if needed here
            # e.g. Adam, RMSprop, or momentum SGD updates
            
    return beta

import numpy as np
import matplotlib.pyplot as plt
import gradient_descent_ols_and_ridge as gd
import gradient_descent_ridge_version2 as gds

# Generate the dataset
np.random.seed(42)  # For reproducibility
n_samples, n_features = 100, 2
X = np.random.rand(n_samples, n_features)

# Simple binary classification problem
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Add a column of ones to X for the intercept term
X = np.c_[np.ones(n_samples), X]

# Initial guess for beta
initial_beta = np.zeros(X.shape[1])

# Gradient Descent Parameters
learning_rate = 0.1
n_iterations = 1000
lmbda = 0.01  # Regularization strength

# Logistic regression with L2 regularization using SGD
beta_gd = stochastic_gradient_descent_logistic_l2(X, y, initial_beta, learning_rate, n_iterations, lmbda)

# Print results
print(f"Learned beta with L2 regularization (lambda={lmbda}): {beta_gd}")
