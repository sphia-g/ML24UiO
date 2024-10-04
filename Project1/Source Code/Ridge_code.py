import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the Franke function
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - 0.25 * ((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1 * (9*y + 1))
    term3 = 0.5 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25 * ((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

# Generate data points
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
x, y = np.meshgrid(x, y)

# Flatten x and y to create a design matrix
x_flat = x.flatten()
y_flat = y.flatten()

# Compute Franke function and add noise
z = FrankeFunction(x, y)
np.random.seed(42)
noise = np.random.normal(0, 1, z.shape)
z_noisy = z + noise
z_noisy_flat = z_noisy.flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x_flat, y_flat, z_noisy_flat, test_size=0.2, random_state=42)

# Reshape the data to 2D arrays
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Standardize the data using Scikit-learn's StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Function to create a design matrix for polynomial terms up to a given degree
def create_design_matrix(x, y, degree):
    N = len(x)
    num_terms = (degree + 1) * (degree + 2) // 2  # Number of polynomial terms up to the given degree
    X = np.ones((N, num_terms))  # Initialize the design matrix
    index = 1
    for i in range(1, degree+1):
        for j in range(i+1):
            X[:, index] = (x ** (i-j)) * (y ** j)
            index += 1
    return X

# ================== Plot 1: MSE and R2 as functions of lambda ==================

# Define the lambda values to be tested
lambda_values = [0.1, 1, 10, 100, 1000]

# Lists to store MSE and R2 scores, and beta coefficients
mse_train_scores = []
mse_test_scores = []
r2_train_scores = []
r2_test_scores = []
betas_ridge_regression = []

degree = 5  # Set degree to 5 for this plot

# Loop over different values of lambda
for lambda_val in lambda_values:
    # Create the design matrix for the current degree (we use 5th degree polynomial for this example)
    X_train = create_design_matrix(x_train_scaled.flatten(), y_train_scaled.flatten(), degree)
    X_test = create_design_matrix(x_test_scaled.flatten(), y_test_scaled.flatten(), degree)
    
    # Perform Ridge regression using the normal equation
    X_train_T = X_train.T
    identity_matrix = np.eye(X_train.shape[1])
    beta = inv(X_train_T @ X_train + lambda_val * identity_matrix) @ X_train_T @ z_train
    
    # Predict values based on the Ridge model
    z_train_pred = X_train @ beta
    z_test_pred = X_test @ beta
    
    # Compute MSE and R2 scores for both training and test data
    mse_train = mean_squared_error(z_train, z_train_pred)
    mse_test = mean_squared_error(z_test, z_test_pred)
    r2_train = r2_score(z_train, z_train_pred)
    r2_test = r2_score(z_test, z_test_pred)
    
    # Append the scores to the lists
    mse_train_scores.append(mse_train)
    mse_test_scores.append(mse_test)
    r2_train_scores.append(r2_train)
    r2_test_scores.append(r2_test)
    
    
# Cinvert list of beta coefficients into an array for plotting
betas_ridge_regression = np.array(betas_ridge_regression, dtype=object)

# Plot the MSE and R2 as functions of lambda
fig, ax1 = plt.subplots()

# Plot MSE on the left y-axis
ax1.set_xlabel('Lambda')
ax1.set_ylabel('MSE', color='tab:red')
ax1.plot(lambda_values, mse_train_scores, 'o-', color='tab:red', label='MSE Train')
ax1.plot(lambda_values, mse_test_scores, 'x-', color='tab:orange', label='MSE Test')
ax1.set_xscale('log')  # Log scale for lambda
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for R2
ax2 = ax1.twinx()
ax2.set_ylabel('R2 Score', color='tab:blue')
ax2.plot(lambda_values, r2_train_scores, 'o-', color='tab:blue', label='R2 Train')
ax2.plot(lambda_values, r2_test_scores, 'x-', color='tab:cyan', label='R2 Test')
ax2.set_xscale('log')  # Log scale for lambda
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add a title and show the plot
plt.title('Ridge Regression: MSE and R2 Scores as Functions of Lambda (Degree = 5)')
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()


# ================== Plot 2: MSE and R2 as functions of polynomial degree ==================

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Define lambda values
lambda_values = [10**-1, 10**0, 10**1, 10**2, 10**3]

# Define the range of polynomial degrees to test
degree_values = [1, 2, 3, 4, 5]

# Iterate over each lambda value to generate independent plots
for lambda_val in lambda_values:
    # Lists to store MSE and R2 scores for different polynomial degrees
    mse_train_degrees = []
    mse_test_degrees = []
    r2_train_degrees = []
    r2_test_degrees = []

    # Loop over different polynomial degrees
    for degree in degree_values:
        # Create the design matrix for the current degree
        X_train = create_design_matrix(x_train_scaled.flatten(), y_train_scaled.flatten(), degree)
        X_test = create_design_matrix(x_test_scaled.flatten(), y_test_scaled.flatten(), degree)
        
        # Perform Ridge regression using the normal equation
        X_train_T = X_train.T
        identity_matrix = np.eye(X_train.shape[1])
        beta = inv(X_train_T @ X_train + lambda_val * identity_matrix) @ X_train_T @ z_train
        
        # Predict values based on the Ridge model
        z_train_pred = X_train @ beta
        z_test_pred = X_test @ beta
        
        # Compute MSE and R2 scores for both training and test data
        mse_train = mean_squared_error(z_train, z_train_pred)
        mse_test = mean_squared_error(z_test, z_test_pred)
        r2_train = r2_score(z_train, z_train_pred)
        r2_test = r2_score(z_test, z_test_pred)
        
        # Append the scores to the lists
        mse_train_degrees.append(mse_train)
        mse_test_degrees.append(mse_test)
        r2_train_degrees.append(r2_train)
        r2_test_degrees.append(r2_test)

    # Create the plot for the current lambda value
    fig, ax1 = plt.subplots()

    # Plot MSE on the left y-axis
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('MSE', color='tab:red')
    ax1.plot(degree_values, mse_train_degrees, 'o-', color='tab:red', label='MSE Train')
    ax1.plot(degree_values, mse_test_degrees, 'x-', color='tab:orange', label='MSE Test')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for R2
    ax2 = ax1.twinx()
    ax2.set_ylabel('R2 Score', color='tab:blue')
    ax2.plot(degree_values, r2_train_degrees, 'o-', color='tab:blue', label='R2 Train')
    ax2.plot(degree_values, r2_test_degrees, 'x-', color='tab:cyan', label='R2 Test')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Add a title and show the plot
    plt.title(f'Ridge Regression: MSE and R2 Scores as Functions of Polynomial Degree (Lambda = {lambda_val})')
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()
