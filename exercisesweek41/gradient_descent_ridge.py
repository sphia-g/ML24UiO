def plain_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda):
    """
    Performs plain gradient descent for Ridge Regression.

    Parameters:
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target vector of shape (n_samples, 1).
    beta : np.ndarray
        Initial coefficients vector (weights) of shape (n_features, 1).
    learning_rate : float
        The step size for each iteration.
    n_iterations : int
        The number of iterations to perform.
    lmbda : float
        The Ridge regularization parameter (L2 penalty).

    Returns:
    beta : np.ndarray
        Updated coefficients after gradient descent.
    """
    
    n = len(X)  # Number of samples

    for iter in range(n_iterations):
        # Compute the gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta
        
        # Update beta
        beta -= learning_rate * gradient

    return beta

def stochastic_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda, batch_size=1):
    """
    Performs stochastic gradient descent for Ridge Regression.

    Parameters:
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target vector of shape (n_samples, 1).
    beta : np.ndarray
        Initial coefficients vector (weights) of shape (n_features, 1).
    learning_rate : float
        The step size for each iteration.
    n_iterations : int
        The number of iterations to perform.
    lmbda : float
        The Ridge regularization parameter (L2 penalty).
    batch_size : int
        The size of the mini-batch to use for each stochastic update (default is 1 for standard SGD).

    Returns:
    beta : np.ndarray
        Updated coefficients after stochastic gradient descent.
    """
    
    n = len(X)  # Number of samples

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
            
            # Compute gradient for the mini-batch with Ridge regularization
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta
            
            # Update beta
            beta -= learning_rate * gradient

    return beta

def momentum_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9):
    """
    Performs gradient descent with momentum for Ridge Regression.

    Parameters:
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target vector of shape (n_samples, 1).
    beta : np.ndarray
        Initial coefficients vector (weights) of shape (n_features, 1).
    learning_rate : float
        The step size for each iteration.
    n_iterations : int
        The number of iterations to perform.
    lmbda : float
        The Ridge regularization parameter (L2 penalty).
    gamma : float
        The momentum factor (default is 0.9).

    Returns:
    beta : np.ndarray
        Updated coefficients after gradient descent with momentum.
    """
    
    n = len(X)  # Number of samples
    velocity = np.zeros_like(beta)  # Initialize velocity (same shape as beta)

    for iter in range(n_iterations):
        # Compute the gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta
        
        # Update the velocity with momentum
        velocity = gamma * velocity + learning_rate * gradient
        
        # Update beta using the velocity
        beta -= velocity

    return beta

def stochastic_gradient_descent_momentum_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9, batch_size=1):
    """
    Stochastic Gradient Descent with Momentum for Ridge Regression.

    Parameters:
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target vector of shape (n_samples, 1).
    beta : np.ndarray
        Initial coefficients vector (weights) of shape (n_features, 1).
    learning_rate : float
        The step size for each iteration.
    n_iterations : int
        The number of iterations to perform.
    lmbda : float
        The Ridge regularization parameter (L2 penalty).
    gamma : float
        The momentum factor (default is 0.9).
    batch_size : int
        The size of the mini-batch to use for each stochastic update (default is 1 for standard SGD).

    Returns:
    beta : np.ndarray
        Updated coefficients after stochastic gradient descent with momentum.
    """
    
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
            
            # Compute gradient for the mini-batch with Ridge regularization
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta
            
            # Update the velocity with momentum
            velocity = gamma * velocity + learning_rate * gradient
            
            # Update beta using velocity
            beta -= velocity

    return beta

def adagrad_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda, epsilon=1e-8):
    """
    Performs Adagrad gradient descent for Ridge Regression.

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
    lmbda : float
        The Ridge regularization parameter (L2 penalty).
    epsilon : float
        A small constant to prevent division by zero (default is 1e-8).

    Returns:
    beta : np.ndarray
        Updated coefficients after Adagrad gradient descent.
    """

    n = len(X)  # Number of samples
    G = np.zeros_like(beta)  # Initialize sum of squared gradients to zero

    for iter in range(n_iterations):
        # Compute gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

        # Accumulate the squared gradient
        G += gradient ** 2

        # Adagrad update: update beta with adaptive learning rate
        beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient

    return beta
