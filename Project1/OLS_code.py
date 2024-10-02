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


# Lists to store MSE, R2 scores, and beta coefficients
degrees = list(range(1, 6))  # Degrees from 1 to 5
mse_train_scores = []
mse_test_scores = []
r2_train_scores = []
r2_test_scores = []
betas = []  # List to store the beta coefficients for each degree

# Loop over different polynomial degrees
for degree in degrees:
    # Create the design matrix for the current degree for training and testing data
    X_train = create_design_matrix(x_train_scaled.flatten(), y_train_scaled.flatten(), degree)
    X_test = create_design_matrix(x_test_scaled.flatten(), y_test_scaled.flatten(), degree)
    
    # Perform the OLS regression using the normal equation
    X_train_T = X_train.T
    beta = inv(X_train_T @ X_train) @ X_train_T @ z_train
    
    # Predict values based on the OLS model
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
    
    # Store the beta coefficients
    betas.append(beta)

