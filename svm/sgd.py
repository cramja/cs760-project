import numpy as np


def sgd_step(w: np.ndarray, x: np.ndarray, y: float, delta: float):
    gradient: np.ndarray = delta * w
    if np.dot(w, x) * y < 1.0:
        gradient -= y * x

    return gradient


def sgd(x: np.ndarray, y: np.ndarray, epoch: int=1, learning_rate: float=0.1):
    x_rows, x_cols = x.shape

    # Start with 0 vector
    w = np.zeros((x_rows,))

    for round in range(epoch):
        print("Round {}".format(round + 1))
        for i in range(x_rows):
            x_i = x[i, :]
            y_i = y[i]

            w -= learning_rate * sgd_step(w, x_i, y_i, delta=0.1)
        print(w)
    return w


def sag(x: np.ndarray, y: np.ndarray, epoch: int=1, learning_rate: float=0.1):
    x_rows, x_cols = x.shape

    # Start with 0 vector
    w = np.zeros((x_rows,))

    acc = 0.0
    last_gradient = 0.0
    for round in range(epoch):
        print("Round {}".format(round + 1))
        for i in range(x_rows):
            x_i = x[i, :]
            y_i = y[i]

            gradient = sgd_step(w, x_i, y_i, delta=0.1)
            acc = acc - last_gradient + gradient
            last_gradient = gradient
            w -= (learning_rate / x_rows) * acc
        print(w)
    return w


def saga(x: np.ndarray, y: np.ndarray, epoch: int=1, learning_rate: float=0.1):
    x_rows, x_cols = x.shape

    # Start with 0 vector
    w = np.zeros((x_rows,))

    average_gradient = np.zeros(w.shape)
    all_gradients = [np.zeros(w.shape) for _ in range(x_rows)]
    for round in range(epoch):
        print("Round {}".format(round + 1))
        for i in range(x_rows):
            x_i = x[i, :]
            y_i = y[i]

            temp_new = w
            gradient = sgd_step(temp_new, x_i, y_i, delta=0.1)
            old_gradient = all_gradients[i]

            # Update weight for new iteration
            w -= learning_rate * (gradient - old_gradient + average_gradient)

            # Update average and table for new iteration
            average_gradient += (gradient - old_gradient) / x_rows
            all_gradients[i] = gradient
        print(w)
    return w


def svrg(x: np.ndarray, y: np.ndarray, epoch: int=1, learning_rate: float=0.1):
    x_rows, x_cols = x.shape

    # Start with 0 vector
    w = np.zeros((x_rows,))

    for round in range(epoch):
        print("Round {}".format(round + 1))

        w_new = w
        sum_gradients = np.zeros((x_rows,))
        for i in range(x_rows):
            x_i = x[i, :]
            y_i = y[i]
            sum_gradients += sgd_step(w_new, x_i, y_i, delta=0.1)
        average_gradients = sum_gradients / x_rows
        w_t = w_new
        for t in range(x_rows):
            x_t = x[t, :]
            y_t = y[t]
            w_t -= learning_rate * (sgd_step(w_t, x_t, y_t, delta=0.1)
                                    - sgd_step(w_new, x_t, y_t, delta=0.1)
                                    + average_gradients)
        w = w_t
        print(w)
    return w


def central_vr(x: np.ndarray, y: np.ndarray, epoch: int=1, learning_rate: float=0.1):
    x_rows, x_cols = x.shape

    # Start with 0 vector
    w = np.zeros((x_rows,))

    all_gradients = [np.zeros(w.shape) for _ in range(x_rows)]
    average_gradient = np.zeros(w.shape)

    # TODO: Normally weight, gradients, average must be initialized with 1 epoch of SGD.

    for round in range(epoch):
        print("Round {}".format(round + 1))
        accumulate = np.zeros(w.shape)
        for i in range(x_rows):
            x_i = x[i, :]
            y_i = y[i]

            old_gradient = all_gradients[i]
            new_gradient = sgd_step(w, x_i, y_i, delta=0.1)

            w -= learning_rate * (new_gradient - old_gradient + average_gradient)
            accumulate += new_gradient / x_rows
            all_gradients[i] = new_gradient

        average_gradient = accumulate
        print(w)

    return w

if __name__ == "__main__":
    x = np.diag([i for i in range(1, 6)])
    y = np.random.choice([-1, 1], (5,))
    print(x)
    print(y)

    w = sgd(x, y, epoch=1000)
    print("W")
    print(w)
    print(np.dot(x, w))
    print(y)

    w = sag(x, y, epoch=1000)
    print("W")
    print(w)
    print(np.dot(x, w))
    print(y)

    w = saga(x, y, epoch=1000)
    print("W")
    print(w)
    print(np.dot(x, w))
    print(y)

    w = svrg(x, y, epoch=1000)
    print("W")
    print(w)
    print(np.dot(x, w))
    print(y)

    w = central_vr(x, y, epoch=1000)
    print("W")
    print(w)
    print(np.dot(x, w))
    print(y)