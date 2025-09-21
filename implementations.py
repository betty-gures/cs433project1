import numpy as np

def compute_gradient(y, tx, w, type="mse", stochastic=False):
    if stochastic:
        random_index = np.random.randint(y.shape[0])
        tx = tx[random_index, :].reshape(1, -1)
        y = y[random_index].reshape(1, )

    e = y - tx @ w
    return - 1/y.shape[0] * tx.T @ e
    

def gradient_descent(y, tx, initial_w, max_iters, gamma, gradient_function, return_history, verbose):
    """The Gradient Descent (GD) algorithm.

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
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        e = y - tx @ w # error
        loss = (e.T @ e) / (2 * y.shape[0])
        losses.append(loss)

        gradient = gradient_function(y, tx, w)
        if verbose: print("Gradient: ", gradient)
        
        # update gradient
        w = w - gamma * gradient

        if verbose: print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    # if max_iters is 0, we still want to return the initial loss
    e = y - tx @ w
    loss = (e.T @ e) / (2 * y.shape[0])
    return w, losses if return_history else loss

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, return_history=False, verbose=False):
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
    gradient_function = lambda y, tx, w: compute_gradient(y, tx, w, type="mse", stochastic=False)
    return gradient_descent(y, tx, initial_w, max_iters, gamma, gradient_function, return_history, verbose)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, return_history=False, verbose=False):
    """The Stochastic Gradient Descent (SGD) algorithm with MSE loss.

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
    gradient_function = lambda y, tx, w: compute_gradient(y, tx, w, type="mse", stochastic=True)
    return gradient_descent(y, tx, initial_w, max_iters, gamma, gradient_function, return_history, verbose)