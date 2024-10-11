from exercisesweek41.gradient_descent_ridge_version2 import adam_gradient_descent_ridge, adam_momentum_gradient_descent_ridge, adam_stochastic_gradient_descent_momentum_ridge, adam_stochastic_gradient_descent_ridge
from gradient_descent_ridge import plain_gradient_descent_ridge, stochastic_gradient_descent_ridge, momentum_gradient_descent_ridge, stochastic_gradient_descent_momentum_ridge, adagrad_gradient_descent_ridge, plain_gradient_descent_adagrad_ridge, stochastic_gradient_descent_adagrad_ridge, momentum_gradient_descent_adagrad_ridge, stochastic_gradient_descent_momentum_adagrad_ridge, plain_gradient_descent_rmsprop_ridge, stochastic_gradient_descent_rmsprop_ridge, momentum_gradient_descent_rmsprop_ridge, stochastic_gradient_descent_momentum_rmsprop_ridge

class RidgeRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000, lmbda=0.001, method="plain", batch_size=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lmbda = lmbda
        self.beta = None
        self.method = method
        self.batch_size = batch_size

    def fit(self, X, y):
        n = len(X)
        self.beta = np.random.randn(X.shape[1], 1)

        # Choose the gradient descent method
        if self.method == "plain":
            self.beta = plain_gradient_descent_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda)
        elif self.method == "momentum":
            self.beta = momentum_gradient_descent_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma)
        elif self.method == "sgd_momentum":
            self.beta = stochastic_gradient_descent_momentum_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.batch_size)
        elif self.method == "stochastic":
            self.beta = stochastic_gradient_descent_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.batch_size)
        elif self.method == "stochastic_adagrad":
            self.beta = stochastic_gradient_descent_adagrad_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.batch_size, self.adagrad, self.epsilon)
        elif self.method == "momentum_adagrad":
            self.beta = momentum_gradient_descent_adagrad_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.adagrad, self.epsilon)
        elif self.method == "sgd_momentum_adagrad":
            self.beta = stochastic_gradient_descent_momentum_adagrad_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.batch_size, self.adagrad, self.epsilon)
        elif self.method == "plain_rmsprop":
            self.beta = plain_gradient_descent_rmsprop_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.gamma, self.rmsprop, self.epsilon)
        elif self.method == "stochastic_rmsprop":
            self.beta = stochastic_gradient_descent_rmsprop_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.gamma, self.rmsprop, self.epsilon)
        elif self.method == "momentum_rmsprop":
            self.beta = momentum_gradient_descent_rmsprop_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.rmsprop, self.epsilon)
        elif self.method == "sgd_momentum_rmsprop":
            self.beta = stochastic_gradient_descent_momentum_rmsprop_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.batch_size, self.rmsprop, self.epsilon)
        elif self.method == "plain" and self.optimizer == "adam":
            self.beta = adam_gradient_descent_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.beta1, self.beta2, self.epsilon)
        elif self.method == "stochastic" and self.optimizer == "adam":
            self.beta = adam_stochastic_gradient_descent_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.beta1, self.beta2, self.epsilon)
        elif self.method == "momentum" and self.optimizer == "adam":
            self.beta = adam_momentum_gradient_descent_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.beta1, self.beta2, self.epsilon)
        elif self.method == "sgd_momentum" and self.optimizer == "adam":
            self.beta = adam_stochastic_gradient_descent_momentum_ridge(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.beta1, self.beta2, self.epsilon)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X):
        return X @ self.beta

    def plot_results(self, X, y, beta_linreg):
        ypredict = X @ self.beta
        ypredict2 = X @ beta_linreg

        plt.plot(X[:, 1], ypredict, "r-", label="Gradient Descent")
        plt.plot(X[:, 1], ypredict2, "b-", label="Analytical Solution")
        plt.plot(X[:, 1], y, 'ro')
        plt.axis([0, 2.0, 0, 15.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Gradient Descent with Ridge Regression')
        plt.legend()
        plt.show()
