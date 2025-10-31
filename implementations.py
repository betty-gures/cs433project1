import numpy as np


def compute_mse(y, tx, w):
    """Calculate the MSE loss.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx @ w
    return (e.T @ e) / (2 * y.shape[0])


def compute_mse_gradient(y, tx, w, stochastic=False):
    """Compute the gradient of the MSE loss.
    
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
        stochastic: boolean, whether to use stochastic gradient (True) or full gradient (False)

    Returns:
        gradient: numpy array of shape (D,), the gradient of the MSE loss at w
    """
    if stochastic:
        random_index = np.random.randint(y.shape[0])
        tx = tx[random_index, :].reshape(1, -1)
        y = y[random_index].reshape(
            1,
        )

    e = y - tx @ w
    return -1 / y.shape[0] * tx.T @ e


def gradient_descent(
    y, tx, initial_w, max_iters, gamma, gradient_function, return_history, verbose
):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: target, numpy array of shape=(N, )
        tx: features, numpy array of shape=(N,2)
        initial_w: weights to start with, numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_mse(y, tx, w)
        losses.append(loss)

        gradient = gradient_function(y, tx, w)
        if verbose:
            print("Gradient: ", gradient)
        # update gradient
        w = w - gamma * gradient

        if verbose:
            print(
                "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    
    loss = compute_mse(y, tx, w) # compute final loss after last gradient update
    return w, losses if return_history else loss


def mean_squared_error_gd(
    y, tx, initial_w, max_iters, gamma, return_history=False, verbose=False
):
    """The Gradient Descent (GD) algorithm with MSE loss.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    gradient_function = lambda y, tx, w: compute_mse_gradient(
        y, tx, w, stochastic=False
    )
    return gradient_descent(
        y, tx, initial_w, max_iters, gamma, gradient_function, return_history, verbose
    )


def mean_squared_error_sgd(
    y, tx, initial_w, max_iters, gamma, return_history=False, verbose=False
):
    """The *Stochastic* Gradient Descent (SGD) algorithm with MSE loss.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of SGD (as well as the final weights)
    """
    gradient_function = lambda y, tx, w: compute_mse_gradient(
        y, tx, w, stochastic=True
    )
    return gradient_descent(
        y, tx, initial_w, max_iters, gamma, gradient_function, return_history, verbose
    )


def least_squares(y, tx):
    """Ordinary Least Squares: Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y) # w = (X^T X)^(-1) X^T y
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression: Linear least squares with L2 regularization.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    gram_matrix = tx.T @ tx  # X^T X (DxD)
    diag_reg = (
        2 * lambda_ * tx.shape[0] * np.eye(tx.shape[1])
    )  # 2 * N * lambda * I (DxD)
    w = np.linalg.solve(
        gram_matrix + diag_reg, tx.T @ y
    )  # w = (X^T X + 2 N lambda I)^(-1) X^T y
    loss = compute_mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """Apply sigmoid function on t (Numerically stable version using different formulas for positive and negative t)

    Args:
        t: logits, scalar or numpy array

    Returns:
        scalar or numpy array
    """

    t = np.asarray(t, dtype=np.float64) # convert to np array if needed
    out = np.empty_like(t)
    pos = t >= 0
    out[pos] = 1 / (1 + np.exp(-t[pos])) # for positive t: 1 / (1 + exp(-t))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1 + exp_t) # for negative t: exp(t) / (1 + exp(t))
    return out


def logistic_regression(y, tx, initial_w, max_iters, gamma, stopping_threshold=1e-8):
    """Logistic regression using gradient descent, assuming y in {0, 1}.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,)
        max_iters: int
        gamma: float

    Returns:
        w: numpy array of shape (D,)
        loss: float
    """
    w = initial_w

    pred = lambda tx, w: sigmoid(tx @ w)  # shape=(N, 1)
    loss = lambda y_pred: -np.mean((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    y_pred = pred(tx, w)
    losses = [loss(y_pred)]

    for _ in range(max_iters):
        gradient = tx.T @ (y_pred - y) / y.shape[0]  # shape=(D,)
        w_new = w - gamma * gradient  # shape=(D,)
        y_pred = pred(tx, w_new)  # shape=(N, 1)
        loss_new = loss(y_pred)

        # if the improvement is below the threshold, stop
        if np.abs(losses[-1] - loss_new) < stopping_threshold:
            break
        losses.append(loss_new)
        w = w_new

    return w, losses[-1]


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, stopping_threshold=1e-8
):
    """L2 Regularized logistic regression using gradient descent, assuming y in {0, 1}.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        lambda_: float
        initial_w: numpy array of shape (D,)
        max_iters: int
        gamma: float
    Returns:
        w: numpy array of shape (D,)
        loss: float
    """
    w = initial_w

    pred = lambda tx, w: sigmoid(tx @ w)  # shape=(N, 1)
    loss = lambda y_pred, w, reg: -np.mean(
        (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    ) + (lambda_ * (w.T @ w) if reg else 0)

    y_pred = pred(tx, w)
    losses = [loss(y_pred, w, True)]

    for _ in range(max_iters):
        gradient = tx.T @ (y_pred - y) / y.shape[0] + 2 * lambda_ * w  # shape=(D,)
        w_new = w - gamma * gradient  # shape=(D,)
        y_pred = pred(tx, w_new)  # shape=(N, 1)
        loss_new = loss(y_pred, w_new, True)

        # early stopping: if the improvement is below the threshold, stop
        if np.abs(losses[-1] - loss_new) < stopping_threshold:
            break
        losses.append(loss_new)
        w = w_new

    return w, loss(pred(tx, w), w, False)
