import numpy as np
import matplotlib.pyplot as plt
import gradient_descent as gd

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
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))  # Binary cross-entropy loss, log likelihood/ loss? is that the same?
    return cost

def logistic_gradient(batch_size, Xi, yi, beta, lmbda=0.001):
    """
    Calculate the gradient for logistic regression with L2 regularization.
    
    Parameters:
    batch_size : int, number of samples in the batch
    Xi : numpy array of shape (batch_size, n), the batch of input data
    yi : numpy array of shape (batch_size,), true labels for the batch
    beta : numpy array of shape (n,), current model weights
    lmbda : float, regularization strength (default: 0.001)
    
    Returns:
    gradient : numpy array of shape (n,), the gradient with respect to beta
    """
    # Predictions with sigmoid
    probabilities = sigmoid(np.dot(Xi, beta))
    
    # Compute the gradient
    gradient = (1 / batch_size) * np.dot(Xi.T, (probabilities - yi))
    
    # Add L2 regularization (do not regularize the intercept term)
    gradient[1:] += (lmbda / batch_size) * beta[1:]
    
    return gradient

def predict(X, beta):
    """
    Generate class predictions (0 or 1) for logistic regression.
    
    Parameters:
    X : numpy array of shape (m, n), the design matrix
    beta : numpy array of shape (n,), the model weights including intercept
    
    Returns:
    predictions : numpy array of shape (m,), predicted class labels (0 or 1)
    """
    # Compute the probability estimates
    probabilities = sigmoid(np.dot(X, beta))
    
    # Convert probabilities to class labels (0 or 1) based on threshold 0.5
    predictions = (probabilities >= 0.5).astype(int)
    return predictions

def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    
    Parameters:
    y_true : numpy array of shape (m,), true labels
    y_pred : numpy array of shape (m,), predicted labels
    
    Returns:
    accuracy : float, the accuracy of predictions
    """
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy