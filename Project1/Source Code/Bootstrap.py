import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.linalg import inv

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
    for i in range(1, degree+1):
        for j in range(i+1):
            X[:, index] = (x ** (i-j)) * (y ** j)
            index += 1
    return X

# Implement Bootstrap Resampling
def bootstrap_resampling(X, z, num_bootstrap_samples, lambda_val, model):
    n,m = X.shape
    mse_bootstrap = []
    betas = np.empty((k, m))

    for _ in range(num_bootstrap_samples):
        # Generate random indices with replacement
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        oob_indices = np.setdiff1d(np.arange(n), bootstrap_indices)

        # Bootstrap training data
        X_train = X[bootstrap_indices]
        z_train = z[bootstrap_indices]

        
        # Out-of-bag test data
        X_test = X[oob_indices]
        z_test = z[oob_indices] 
        
        if model == "Lasso":
            # Fit the Lasso model
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
        # Predict and calculate MSE for the OOB samples
        mse_test = mean_squared_error(z_test, z_test_pred)
        mse_bootstrap.append(mse_test)

    return np.mean(mse_bootstrap)  # Return the average MSE across bootstrap samples

# Parameters
lambda_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5, 10, 100, 1000]
degree = 5  # Set degree to 5 for this plot
num_bootstrap_samples = 100  # Number of bootstrap samples
k = 5
model = "OLS"

# Create design matrix for the current degree
X_scaled = create_design_matrix(x_scaled.flatten(), y_scaled.flatten(), degree)

# Store MSE results for each lambda
mse_bootstrap_scores = []


for lambda_val in lambda_values:
    # Perform bootstrap resampling and store MSE
    mse_bootstrap = bootstrap_resampling(X_scaled, z_noisy_flat, num_bootstrap_samples, lambda_val, model)
    mse_bootstrap_scores.append(mse_bootstrap)


# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(lambda_values, mse_bootstrap_scores, 'o-', color='tab:red', label='%s' %model)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Lasso Regression: MSE with Bootstrap Resampling')
plt.legend()
plt.grid(True)
plt.show()