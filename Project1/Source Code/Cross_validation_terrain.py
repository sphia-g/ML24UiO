import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

# Define the Franke function
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - 0.25 * ((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1 * (9*y + 1))
    term3 = 0.5 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25 * ((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
# Use data points from TIFF file
terrain1_array = np.array(terrain1)
# Get image dimensions
height, width = terrain1_array.shape[:2]
# Create meshgrid for X and Y coordinates
x, y = np.meshgrid(range(width), range(height))

# Flatten x and y to create a design matrix
x_flat = x.flatten()
y_flat = y.flatten()

# Compute Franke function and add noise
z = FrankeFunction(x, y)
np.random.seed(42)
noise = np.random.normal(0, 1, z.shape)
z_noisy = z + noise
z_noisy_flat = z_noisy.flatten()

# Standardize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x_flat.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(y_flat.reshape(-1, 1))


# Function to create a design matrix for polynomial terms up to a given degree
def create_design_matrix(x, y, degree):
    N = len(x)
    num_terms = (degree + 1) * (degree + 2) // 2  # Number of polynomial terms up to the given degree
    X = np.ones((N, num_terms))  # Initialize the design matrix
    index = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, index] = (x ** (i - j)) * (y ** j)
            index += 1
    return X

# Implement k-fold cross-validation
def k_fold_cross_validation(X, z, k, lambda_val, model):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_folds = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index] 

        if model == "Lasso":        
            lasso_model = Lasso(alpha=lambda_val, max_iter=10000)
            lasso_model.fit(X_train, z_train)
            z_test_pred = lasso_model.predict(X_test)
        elif model == "Ridge":
            X_train_T = X_train.T
            identity_matrix = np.eye(X_train.shape[1])
            beta = inv(X_train_T @ X_train + lambda_val * identity_matrix) @ X_train_T @ z_train
            z_test_pred = X_test @ beta
        elif model == "OLS":
            X_train_T = X_train.T
            beta = inv(X_train_T @ X_train) @ X_train_T @ z_train
            z_test_pred = X_test @ beta
        else:
            raise Exception("does not recognize model")

        # Predict and calculate MSE for the test fold
        mse_test = mean_squared_error(z_test, z_test_pred)
        mse_folds.append(mse_test)

    return np.mean(mse_folds), np.std(mse_folds)  # Return the average MSE and its standard deviation

# Define the polynomial degrees to test
degrees = range(1, 6)  # Change as needed to test degrees from 1 to 10
k = 5  # Number of bootstrap samples
model = "Lasso"
lambda_degree = 0.1

# Store results in a list of dictionaries for easier DataFrame conversion
results = []

# Loop over the polynomial degrees
for degree in degrees:
    # Create the design matrix for the current degree
    X_scaled = create_design_matrix(x_scaled.flatten(), y_scaled.flatten(), degree)
    
    # Perform bootstrap resampling to get mean MSE and standard deviation for OLS
    mean_mse, std_mse = k_fold_cross_validation(X_scaled, z_noisy_flat, k, 0.1, model)
    
    # Store the results in a dictionary
    results.append({
        "Degree": degree,
        "Mean MSE": mean_mse,
        "Standard Deviation": std_mse
    })

# Convert the results to a pandas DataFrame for a table-like output
results_df = pd.DataFrame(results)

# Display the table
print(results_df)

