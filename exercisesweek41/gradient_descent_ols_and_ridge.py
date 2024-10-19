import numpy as np

# Gradient descent : plain and gradient
# regression models : ols and ridge
# with and with out momentum
# approximations = ADAM ADAgrad RMS

def plain_gradient_descent(X, y, beta, learning_rate, n_iterations, model=None, approx = None, momentum=False, gamma=0.9, momentum_gamma=0.999, epsilon=1e-8, lmbda=0.001):

    n = len(X)
    G = np.zeros_like(beta)
    velocity = np.zeros_like(beta)
    t = 0 ## only for ADAM

    #if approx != None and approx is adam:
    #    momentum_gamma = 0.999

    for _ in range(n_iterations):
        t+=1 ## only for ADAM

        gradient = model(n, X, y, beta, lmbda)
        ##gradient = (2.0 / n) * X.T @ (X @ beta - y)

        if approx is not None:
            G, beta = approx(G, momentum, velocity, gradient, learning_rate, epsilon, gamma, momentum_gamma, beta, t)
        elif momentum:
            velocity = gamma * velocity + learning_rate * gradient
            beta -= velocity 
        else:
            beta -= learning_rate * gradient

    return beta

# if not momentum, let batchsize be the default 1
def stochastic_gradient_descent(X, y, beta, learning_rate, n_iterations, model=None, approx=None, momentum=False, gamma=0.9, momentum_gamma=0.999, epsilon=1e-8, batch_size=1, lmbda = 0.001):
    n = len(X)
    velocity = np.zeros_like(beta)  # Initialize velocity (same shape as beta)  
    G = np.zeros_like(beta)  # Initialize sum of squared gradients for Adagrad  
    t=0 ##only for ADAM

    for _ in range(n_iterations):
        if approx is not None or momentum: 
            indices = np.random.permutation(n)
            X = X[indices]
            y = y[indices]

        for i in range(0, n, batch_size):
            t+=1
            Xi = X[i:i+batch_size]
            yi = y[i:i+batch_size]

            gradient = model(batch_size, Xi, yi, beta, lmbda)
                ##gradient = (2.0 / batch_size) * Xi.T @ (Xi @ beta - yi) 

            if approx is not None:
                G, beta = approx(G, momentum, velocity, gradient, learning_rate, epsilon, gamma, momentum_gamma, beta, t)
            elif momentum:
                velocity = gamma * velocity + learning_rate * gradient
                beta -= velocity
            else:
                #gradient = 2.0 * Xi.T @ (Xi @ beta - yi)
                beta -= learning_rate * gradient
    return beta

def adagrad(G, momentum, velocity, gradient, learning_rate, epsilon, gamma, momentum_gamma, beta, t):
    G = G+ gradient**2 
    if momentum:    
        adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
        velocity = gamma * velocity + adaptive_lr * gradient
        beta -= velocity
    else:
        beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
    return G, beta


def rmsprop(G, momentum, velocity, gradient, learning_rate, epsilon, gamma, momentum_gamma, beta, t):
    
    G = gamma * G + (1 - gamma) * gradient**2

    if momentum:
        adaptive_lr = learning_rate / (np.sqrt(G + epsilon))
        velocity = momentum_gamma * velocity + adaptive_lr * gradient
        beta -= velocity
    else:
        beta -= (learning_rate / (np.sqrt(G + epsilon))) * gradient
    return G, beta

def adam(G, momentum, velocity, gradient, learning_rate, epsilon, gamma, momentum_gamma, beta, t): 
    ## no momentum implementation atm
    beta1 = gamma ## is this good code??
    beta2 = momentum_gamma
    m = G
        # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * gradient
        # Update biased second moment estimate
    velocity = beta2 * velocity + (1 - beta2) * (gradient ** 2)

        # Compute bias-corrected first and second moment estimates
    m_hat = m / (1 - beta1**t)
    v_hat = velocity / (1 - beta2**t)
    beta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return m, beta

def ols(n, X, y, beta):
    gradient = (2.0 / n)* X.T @ (X @ beta - y)
    return gradient

def ridge(n, X, y, beta, lmbda):
    gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta

    return gradient



"""
My reasoning for not placing stochastic and plain in the same function: 
because of the nested for-loop in stochastic
(would make momentum etc.. very difficult and messy)
"""