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

def plain_gradient_descent_adagrad(X, y, beta, learning_rate, n_iterations, adagrad=False, epsilon=1e-8):
    """
    Performs plain gradient descent with optional Adagrad.

    Parameters:
    adagrad : bool
        If True, Adagrad is applied to tune the learning rate adaptively.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)
        
        if adagrad:
            # Accumulate the squared gradient
            G += gradient ** 2
            # Adagrad update
            beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
        else:
            # Standard GD update
            beta -= learning_rate * gradient

    return beta

def stochastic_gradient_descent_adagrad(X, y, beta, learning_rate, n_iterations, batch_size=1, adagrad=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with optional Adagrad.
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

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

            if adagrad:
                # Accumulate the squared gradient
                G += gradient ** 2
                # Adagrad update
                beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
            else:
                # Standard SGD update
                beta -= learning_rate * gradient

    return beta

def momentum_gradient_descent_adagrad(X, y, beta, learning_rate, n_iterations, gamma=0.9, adagrad=False, epsilon=1e-8):
    """
    Performs gradient descent with momentum and optional Adagrad.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

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

def stochastic_gradient_descent_momentum_adagrad(X, y, beta, learning_rate, n_iterations, gamma=0.9, batch_size=1, adagrad=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum and optional Adagrad.
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

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

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

def plain_gradient_descent_rmsprop(X, y, beta, learning_rate, n_iterations, gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs plain gradient descent with optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

        if rmsprop:
            # Update the running average of squared gradients
            G = gamma * G + (1 - gamma) * gradient ** 2
            # RMSprop update
            beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
        else:
            # Standard GD update
            beta -= learning_rate * gradient

    return beta

def stochastic_gradient_descent_rmsprop(X, y, beta, learning_rate, n_iterations, batch_size=1, gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with optional RMSprop.
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

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

            if rmsprop:
                # Update the running average of squared gradients
                G = gamma * G + (1 - gamma) * gradient ** 2
                # RMSprop update
                beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
            else:
                # Standard SGD update
                beta -= learning_rate * gradient

    return beta

def momentum_gradient_descent_rmsprop(X, y, beta, learning_rate, n_iterations, gamma=0.9, momentum_gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs gradient descent with momentum and optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

        if rmsprop:
            # Update the running average of squared gradients
            G = gamma * G + (1 - gamma) * gradient ** 2
            # RMSprop update
            adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
            velocity = momentum_gamma * velocity + adaptive_lr * gradient
        else:
            # Standard momentum GD update
            velocity = momentum_gamma * velocity + learning_rate * gradient

        beta -= velocity

    return beta

def stochastic_gradient_descent_momentum_rmsprop(X, y, beta, learning_rate, n_iterations, gamma=0.9, momentum_gamma=0.9, batch_size=1, rmsprop=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum and optional RMSprop.
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

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

            if rmsprop:
                # Update the running average of squared gradients
                G = gamma * G + (1 - gamma) * gradient ** 2
                # RMSprop update
                adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
                velocity = momentum_gamma * velocity + adaptive_lr * gradient
            else:
                # Standard SGD with momentum update
                velocity = momentum_gamma * velocity + learning_rate * gradient

            beta -= velocity

    return beta

def plain_gradient_descent_rmsprop(X, y, beta, learning_rate, n_iterations, gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs plain gradient descent with optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

        if rmsprop:
            # Update the running average of squared gradients
            G = gamma * G + (1 - gamma) * gradient ** 2
            # RMSprop update
            beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
        else:
            # Standard GD update
            beta -= learning_rate * gradient

    return beta

def stochastic_gradient_descent_rmsprop(X, y, beta, learning_rate, n_iterations, batch_size=1, gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with optional RMSprop.
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

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

            if rmsprop:
                # Update the running average of squared gradients
                G = gamma * G + (1 - gamma) * gradient ** 2
                # RMSprop update
                beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
            else:
                # Standard SGD update
                beta -= learning_rate * gradient

    return beta

def momentum_gradient_descent_rmsprop(X, y, beta, learning_rate, n_iterations, gamma=0.9, momentum_gamma=0.9, rmsprop=False, epsilon=1e-8):
    """
    Performs gradient descent with momentum and optional RMSprop.
    """

    n = len(X)
    G = np.zeros_like(beta)  # Initialize running average of squared gradients
    velocity = np.zeros_like(beta)  # Initialize velocity for momentum

    for iter in range(n_iterations):
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

        if rmsprop:
            # Update the running average of squared gradients
            G = gamma * G + (1 - gamma) * gradient ** 2
            # RMSprop update
            adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
            velocity = momentum_gamma * velocity + adaptive_lr * gradient
        else:
            # Standard momentum GD update
            velocity = momentum_gamma * velocity + learning_rate * gradient

        beta -= velocity

    return beta

def stochastic_gradient_descent_momentum_rmsprop(X, y, beta, learning_rate, n_iterations, gamma=0.9, momentum_gamma=0.9, batch_size=1, rmsprop=False, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum and optional RMSprop.
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

            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

            if rmsprop:
                # Update the running average of squared gradients
                G = gamma * G + (1 - gamma) * gradient ** 2
                # RMSprop update
                adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
                velocity = momentum_gamma * velocity + adaptive_lr * gradient
            else:
                # Standard SGD with momentum update
                velocity = momentum_gamma * velocity + learning_rate * gradient

            beta -= velocity

    return beta

def adam_plain_gradient_descent(X, y, beta, learning_rate, n_iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs plain gradient descent with Adam optimization for OLS.
    """
    
    n = len(X)
    m = np.zeros_like(beta)  # Initialize first moment vector
    v = np.zeros_like(beta)  # Initialize second moment vector
    t = 0  # Initialize timestep

    for iter in range(n_iterations):
        t += 1
        # Compute gradient
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

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

def adam_stochastic_gradient_descent(X, y, beta, learning_rate, n_iterations, batch_size=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs stochastic gradient descent with Adam optimization for OLS.
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

            # Compute gradient
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

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

def adam_momentum_gradient_descent(X, y, beta, learning_rate, n_iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs gradient descent with momentum and Adam optimization for OLS.
    """
    
    n = len(X)
    m = np.zeros_like(beta)  # Initialize first moment vector
    v = np.zeros_like(beta)  # Initialize second moment vector
    t = 0  # Initialize timestep

    for iter in range(n_iterations):
        t += 1
        # Compute gradient
        gradient = (2.0 / n) * X.T @ (X @ beta - y)

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

def adam_stochastic_gradient_descent_momentum(X, y, beta, learning_rate, n_iterations, batch_size=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs stochastic gradient descent with momentum and Adam optimization for OLS.
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

            # Compute gradient
            gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi)

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

