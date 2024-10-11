from exercisesweek41.gradient_descent_version2 import adam_momentum_gradient_descent, adam_plain_gradient_descent, adam_stochastic_gradient_descent, adam_stochastic_gradient_descent_momentum
from gradient_descent import plain_gradient_descent, stochastic_gradient_descent, momentum_gradient_descent, stochastic_gradient_descent_momentum, plain_gradient_descent_adagrad, stochastic_gradient_descent_adagrad, momentum_gradient_descent_adagrad, stochastic_gradient_descent_momentum_adagrad, plain_gradient_descent_rmsprop,stochastic_gradient_descent_rmsprop, momentum_gradient_descent_rmsprop, stochastic_gradient_descent_momentum_rmsprop

class OLS:
    def __init__(self, learning_rate=0.0001, n_iterations=1000, method="plain", gamma=0.9, batch_size=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
        self.method = method
        self.gamma = gamma
        self.batch_size = batch_size

    def fit(self, X, y):
        n = len(X)
        self.beta = np.random.randn(X.shape[1], 1)

        # Choose the gradient descent method
        if self.method == "plain":
            self.beta = plain_gradient_descent(X, y, self.beta, self.learning_rate, self.n_iterations)
        elif self.method == "stochastic":
            self.beta = stochastic_gradient_descent(X, y, self.beta, self.learning_rate, self.n_iterations)
        elif self.method == "momentum":
            self.beta = momentum_gradient_descent(X, y, self.beta, self.learning_rate, self.n_iterations, self.gamma)
        elif self.method == "sgd_momentum":
            self.beta = stochastic_gradient_descent_momentum(X, y, self.beta, self.learning_rate, self.n_iterations, self.gamma, self.batch_size)
        elif self.method == "plain_adagrad":
            self.beta = plain_gradient_descent_adagrad(X, y, self.beta, self.learning_rate, self.n_iterations, self.adagrad, self.epsilon)
        elif self.method == "stochastic_adagrad":
            self.beta = stochastic_gradient_descent_adagrad(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.adagrad, self.epsilon)
        elif self.method == "momentum_adagrad":
            self.beta = momentum_gradient_descent_adagrad(X, y, self.beta, self.learning_rate, self.n_iterations, self.gamma, self.adagrad, self.epsilon)
        elif self.method == "sgd_momentum_adagrad":
            self.beta = stochastic_gradient_descent_momentum_adagrad(X, y, self.beta, self.learning_rate, self.n_iterations, self.gamma, self.batch_size, self.adagrad, self.epsilon)
        elif self.method == "plain_rmsprop":
            self.beta = plain_gradient_descent_rmsprop(X, y, self.beta, self.learning_rate, self.n_iterations, self.gamma, self.rmsprop, self.epsilon)
        elif self.method == "stochastic_rmsprop":
            self.beta = stochastic_gradient_descent_rmsprop(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.gamma, self.rmsprop, self.epsilon)
        elif self.method == "momentum_rmsprop":
            self.beta = momentum_gradient_descent_rmsprop(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.rmsprop, self.epsilon)
        elif self.method == "sgd_momentum_rmsprop":
            self.beta = stochastic_gradient_descent_momentum_rmsprop(X, y, self.beta, self.learning_rate, self.n_iterations, self.lmbda, self.gamma, self.batch_size, self.rmsprop, self.epsilon)
        elif self.method == "plain" and self.optimizer == "adam":
            self.beta = adam_plain_gradient_descent(X, y, self.beta, self.learning_rate, self.n_iterations, self.beta1, self.beta2, self.epsilon)
        elif self.method == "stochastic" and self.optimizer == "adam":
            self.beta = adam_stochastic_gradient_descent(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.beta1, self.beta2, self.epsilon)
        elif self.method == "momentum" and self.optimizer == "adam":
            self.beta = adam_momentum_gradient_descent(X, y, self.beta, self.learning_rate, self.n_iterations, self.beta1, self.beta2, self.epsilon)
        elif self.method == "sgd_momentum" and self.optimizer == "adam":
            self.beta = adam_stochastic_gradient_descent_momentum(X, y, self.beta, self.learning_rate, self.n_iterations, self.batch_size, self.beta1, self.beta2, self.epsilon)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X):
        return X @ self.beta

    def plot_results(self, X, y, beta_linreg):
        xnew = np.array([[0], [2]])
        xbnew = np.c_[np.ones((2, 1)), xnew]
        ypredict = xbnew.dot(self.beta)
        ypredict2 = xbnew.dot(beta_linreg)

        plt.plot(xnew, ypredict, "r-", label="Gradient Descent")
        plt.plot(xnew, ypredict2, "b-", label="Analytical Solution")
        plt.plot(X[:, 1], y, 'ro')
        plt.axis([0, 2.0, 0, 15.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Gradient Descent with OLS')
        plt.legend()
        plt.show()
