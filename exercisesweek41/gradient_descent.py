import numpy as np

def plain_gradient_descent(X, y, beta, learning_rate, n_iterations):
    n = len(X)
    for iter in range(n_iterations):
        gradients = (2.0 / n) * X.T @ (X @ beta - y)
        beta -= learning_rate * gradients
    return beta

def stochastic_gradient_descent(X, y, beta, learning_rate, n_iterations):
    n = len(X)
    for iter in range(n_iterations):
        for i in range(n):
            xi = X[i:i+1]
            yi = y[i:i+1]
            gradients = 2.0 * xi.T @ (xi @ beta - yi)
            beta -= learning_rate * gradients
    return beta

def momentum_gradient_descent(X, y, beta, learning_rate, n_iterations, gamma=0.9):
    n = len(X)
    velocity = np.zeros_like(beta)
    
    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)
        
        # Update velocity with momentum
        velocity = gamma * velocity + learning_rate * gradient
        
        # Update beta
        beta -= velocity
    
    return beta


def stochastic_gradient_descent_momentum(X, y, beta, learning_rate, n_iterations, gamma=0.9, batch_size=1):
   
    n = len(X)  # Number of samples
    velocity = np.zeros_like(beta)  # Initialize velocity (same shape as beta)

    for iter in range(n_iterations):
        # Shuffle the data at the beginning of each iteration
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Perform stochastic updates
        for i in range(0, n, batch_size):
            # Select mini-batch
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            
            # Compute gradient for the mini-batch
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)
            
            # Update the velocity with momentum
            velocity = gamma * velocity + learning_rate * gradient
            
            # Update beta using velocity
            beta -= velocity

    return beta

import numpy as np

def adagrad_gradient_descent(X, y, beta, learning_rate, n_iterations, epsilon=1e-8):
    """
    Performs Adagrad gradient descent without momentum.

    Parameters:
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target vector of shape (n_samples, 1).
    beta : np.ndarray
        Initial coefficients vector (weights) of shape (n_features, 1).
    learning_rate : float
        The initial learning rate for Adagrad.
    n_iterations : int
        The number of iterations to perform.
    epsilon : float
        A small constant to prevent division by zero (default is 1e-8).

    Returns:
    beta : np.ndarray
        Updated coefficients after Adagrad gradient descent.
    """

    n = len(X)  # Number of samples
    G = np.zeros_like(beta)  # Initialize sum of squared gradients to zero

    for iter in range(n_iterations):
        # Compute gradient
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

        # Accumulate the squared gradient
        G += gradient ** 2

        # Adagrad update: update beta with adaptive learning rate
        beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient

    return beta
