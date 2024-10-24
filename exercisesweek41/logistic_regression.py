def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_cost(X, y, beta):
    """
    Compute the binary cross-entropy cost for logistic regression.

    Parameters:
    X : numpy array of shape (m, n), the design matrix
    y : numpy array of shape (m,), true labels (0 or 1)
    beta : numpy array of shape (n,), the model weights including intercept
    
    Returns:
    cost : scalar value of the cost
    """
    m = X.shape[0]
    h = sigmoid(np.dot(X, beta))  # Predictions using sigmoid
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))  # Binary cross-entropy loss
    return cost

def compute_logistic_gradient(X, y, beta):
    """
    Compute the gradient for logistic regression.
    
    Parameters:
    X : numpy array of shape (m, n), the design matrix
    y : numpy array of shape (m,), true labels (0 or 1)
    beta : numpy array of shape (n,), the model weights including intercept
    
    Returns:
    gradient : numpy array of shape (n,), the gradient of the cost with respect to beta
    """
    m = X.shape[0]
    h = sigmoid(np.dot(X, beta))  # Predictions using sigmoid
    gradient = (1 / m) * np.dot(X.T, (h - y))  # Gradient of binary cross-entropy
    return gradient

def stochastic_gradient_descent_logistic(X, y, beta, learning_rate, n_iterations, approx=None, momentum=False):
    """
    Stochastic Gradient Descent for Logistic Regression.

    Parameters:
    X : numpy array of shape (m, n), the design matrix
    y : numpy array of shape (m,), true labels (0 or 1)
    beta : numpy array of shape (n,), initial weights (including intercept)
    learning_rate : scalar, learning rate for SGD
    n_iterations : int, number of iterations
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
            
            # Compute the gradient using the current example
            gradient = compute_logistic_gradient(xi, yi, beta)
            
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

# True beta for logistic regression isn't relevant for classification, we can skip that part for now
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple binary classification problem

# Add a column of ones to X for the intercept term
X = np.c_[np.ones(n_samples), X]

# Initial guess for beta
initial_beta = np.zeros(X.shape[1])

# Gradient Descent Parameters
learning_rate = 0.1
n_iterations = 1000

# Logistic regression-specific SGD
beta_gd = stochastic_gradient_descent_logistic(X, y, initial_beta, learning_rate, n_iterations)

# Print results
print(f"Learned beta from logistic regression SGD: {beta_gd}")

# You can also compare against your other methods here if you wish
