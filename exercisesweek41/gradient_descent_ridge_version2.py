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

def plain_gradient_descent_adagrad_ridge(X, y, beta, learning_rate, n_iterations, lmbda, adagrad=False, epsilon=1e-8):
    """
    Performs plain gradient descent for Ridge Regression with optional Adagrad.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta
        
        if adagrad:
            # Accumulate the squared gradient
            G += gradient ** 2
            # Adagrad update
            beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
        else:
            # Standard GD update
            beta -= learning_rate * gradient

    return beta

def stochastic_gradient_descent_adagrad_ridge(X, y, beta, learning_rate, n_iterations, lmbda, batch_size=1, adagrad=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent for Ridge Regression with optional Adagrad.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad

    for iter in range(n_iterations):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta

            if adagrad:
                # Accumulate the squared gradient
                G += gradient ** 2
                # Adagrad update
                beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
            else:
                # Standard SGD update
                beta -= learning_rate * gradient

    return beta

def momentum_gradient_descent_adagrad_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9, adagrad=False, epsilon=1e-8):
    """
    Performs gradient descent with momentum for Ridge Regression with optional Adagrad.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

        if adagrad:
            # Accumulate the squared gradient
            G += gradient ** 2
            # Adagrad learning rate adjustment
            adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
            velocity = gamma * velocity + adaptive_lr * gradient
        else:
            # Standard momentum GD update
            velocity = gamma * velocity + learning_rate * gradient

        beta -= velocity

    return beta

def stochastic_gradient_descent_momentum_adagrad_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9, batch_size=1, adagrad=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum for Ridge Regression with optional Adagrad.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta

            if adagrad:
                # Accumulate the squared gradient
                G += gradient ** 2
                # Adagrad learning rate adjustment
                adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
                velocity = gamma * velocity + adaptive_lr * gradient
            else:
                # Standard SGD with momentum update
                velocity = gamma * velocity + learning_rate * gradient

            beta -= velocity

    return beta

def plain_gradient_descent_rmsprop_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs plain gradient descent for Ridge Regression with optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients

    for iter in range(n_iterations):
        # Compute gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

        if rmsprop:
            # Update the running average of squared gradients
            G = gamma * G + (1 - gamma) * gradient ** 2
            # RMSprop update
            beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
        else:
            # Standard GD update
            beta -= learning_rate * gradient

    return beta

def stochastic_gradient_descent_rmsprop_ridge(X, y, beta, learning_rate, n_iterations, lmbda, batch_size=1, gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent for Ridge Regression with optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients

    for iter in range(n_iterations):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            # Compute gradient with Ridge regularization
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta

            if rmsprop:
                # Update the running average of squared gradients
                G = gamma * G + (1 - gamma) * gradient ** 2
                # RMSprop update
                beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
            else:
                # Standard SGD update
                beta -= learning_rate * gradient

    return beta

def momentum_gradient_descent_rmsprop_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9, momentum_gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs gradient descent with momentum for Ridge Regression with optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        # Compute gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

        if rmsprop:
            # Update the running average of squared gradients
            G = gamma * G + (1 - gamma) * gradient ** 2
            # RMSprop learning rate adjustment
            adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
            velocity = momentum_gamma * velocity + adaptive_lr * gradient
        else:
            # Standard momentum GD update
            velocity = momentum_gamma * velocity + learning_rate * gradient

        beta -= velocity

    return beta

def stochastic_gradient_descent_momentum_rmsprop_ridge(X, y, beta, learning_rate, n_iterations, lmbda, gamma=0.9, momentum_gamma=0.9, batch_size=1, rmsprop=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum for Ridge Regression with optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            # Compute gradient with Ridge regularization
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta

            if rmsprop:
                # Update the running average of squared gradients
                G = gamma * G + (1 - gamma) * gradient ** 2
                # RMSprop learning rate adjustment
                adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
                velocity = momentum_gamma * velocity + adaptive_lr * gradient
            else:
                # Standard SGD with momentum update
                velocity = momentum_gamma * velocity + learning_rate * gradient

            beta -= velocity

    return beta

def adam_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda=0, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs plain gradient descent with Adam optimization.
    """
    
    n = len(X)
    m = np.zeros_like(beta)  # Initialize first moment vector
    v = np.zeros_like(beta)  # Initialize second moment vector
    t = 0  # Initialize timestep

    for iter in range(n_iterations):
        t += 1
        # Compute gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Compute bias-corrected first and second moment estimates
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        beta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return beta

def adam_stochastic_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda=0, batch_size=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs stochastic gradient descent with Adam optimization.
    """
    
    n = len(X)
    m = np.zeros_like(beta)  # Initialize first moment vector
    v = np.zeros_like(beta)  # Initialize second moment vector
    t = 0  # Initialize timestep

    for iter in range(n_iterations):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            t += 1
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            # Compute gradient with Ridge regularization
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update parameters
            beta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return beta

def adam_momentum_gradient_descent_ridge(X, y, beta, learning_rate, n_iterations, lmbda=0, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs gradient descent with momentum and Adam optimization.
    """
    
    n = len(X)
    m = np.zeros_like(beta)  # Initialize first moment vector
    v = np.zeros_like(beta)  # Initialize second moment vector
    t = 0  # Initialize timestep

    for iter in range(n_iterations):
        t += 1
        # Compute gradient with Ridge regularization
        gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Compute bias-corrected first and second moment estimates
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        beta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return beta

def adam_stochastic_gradient_descent_momentum_ridge(X, y, beta, learning_rate, n_iterations, lmbda=0, batch_size=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum and Adam optimization.
    """
    
    n = len(X)
    m = np.zeros_like(beta)  # Initialize first moment vector
    v = np.zeros_like(beta)  # Initialize second moment vector
    t = 0  # Initialize timestep

    for iter in range(n_iterations):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            t += 1
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            # Compute gradient with Ridge regularization
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) + 2 * lmbda * beta

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update parameters
            beta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return beta

