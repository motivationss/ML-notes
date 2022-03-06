# EECS 545 HW3 Q4
# work with Yuxuan Song

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(545)

# Instruction: use these hyperparameters for both (b) and (d)
eta = 0.5
C = 5
iterNums = [5, 50, 100, 1000, 5000, 6000]


def indicator(label, w, i, b, matrix):
    return 1 if np.dot(label[i], np.dot(w.T, matrix[i]) + b) < 1 else 0


def svm_train_bgd(matrix: np.ndarray, label: np.ndarray, nIter: int):
    # Implement your algorithm and return state (e.g., learned model)
    # Implement your algorithm and return state (e.g., learned model)
    state = {}
    N, D = matrix.shape
    label = label.reshape((N, 1))
    iter_curr = 0
    # first, initialize w and b
    w = np.zeros((D, 1))
    b = 0
    while iter_curr < nIter:
        w_grad_star = np.zeros((D, 1))
        b_grad_star = 0
        for i in range(N):
            I = indicator(label, w, i, b, matrix)
            matrix_i = matrix[i].reshape((D, 1))
            w_grad_star += np.dot(I, np.dot(matrix_i, label[i][0]))
            b_grad_star += np.dot(I, label[i][0])
        w_grad = w - C * w_grad_star
        b_grad = -C * b_grad_star
        alpha_i = eta / (1 + iter_curr * eta)
        w = w - alpha_i * w_grad
        b = b - 0.01 * alpha_i * b_grad
        iter_curr += 1
    state['w'] = w
    state['b'] = b

    return state


def svm_train_sgd(matrix: np.ndarray, label: np.ndarray, nIter: int):
    # Implement your algorithm and return state (e.g., learned model)
    state = {}
    N, D = matrix.shape
    label = label.reshape((N, 1))
    iter_curr = 0
    w = np.zeros((D, 1))
    b = 0
    while iter_curr < nIter:
        for i in range(N):
            I = indicator(label, w, i, b, matrix)
            matrix_i = matrix[i].reshape((matrix[i].shape[0], 1))
            w_grad_i = np.dot(I, np.dot(label[i][0], matrix_i))
            b_grad_i = np.dot(I, label[i][0])
            w_grad = w / N - C * w_grad_i
            b_grad = -C * b_grad_i
            alpha_i = eta / (1 + iter_curr * eta)
            w = w - alpha_i * w_grad
            b = b - 0.01 * alpha_i * b_grad
        iter_curr += 1
    state['w'] = w
    state['b'] = b

    return state


def svm_test(matrix: np.ndarray, state):
    # Classify each test data as +1 or -1
    output = np.ones((matrix.shape[0], 1))

    w_matrix = state['w']
    b_matrix = state['b']
    prediction = matrix @ w_matrix + b_matrix

    for i in range(matrix.shape[0]):
        output[i] = 1 if prediction[i] >= 0 else -1

    return output


def evaluate(output: np.ndarray, label: np.ndarray, nIter: int) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    accuracy = (label * output > 0).sum() * 1. / len(output)
    print('[Iter {:4d}: accuracy = {:2.4f}%'.format(nIter, 100 * accuracy))
    print('\n')

    return accuracy


def load_data():
    # Note1: label is {-1, +1}
    # Note2: data matrix shape  = [Ndata, 4]
    # Note3: label matrix shape = [Ndata, 1]

    # Load data
    q4_data = np.load('q4_data/q4_data.npy', allow_pickle=True).item()

    train_x = q4_data['q4x_train']
    train_y = q4_data['q4y_train']
    test_x = q4_data['q4x_test']
    test_y = q4_data['q4y_test']
    return train_x, train_y, test_x, test_y


def run_bgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **batch gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_bgd(train_x, train_y, nIter)
        print("Trained Parameter w: ")
        print(str(state['w']))
        print("Trained Parameter b: ", str(state['b']))

        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter)


def run_sgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **stocahstic gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)

    [Note: Use the same hyperparameters as (b)]
    [Note: If you implement it correctly, the running time will be ~15 sec]
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_sgd(train_x, train_y, nIter)
        print("Trained Parameter w: ")
        print(str(state['w']))
        print("Trained Parameter b: ", str(state['b']))

        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter)


def main():
    train_x, train_y, test_x, test_y = load_data()
    print("q2(c) Batch training: ")
    run_bgd(train_x, train_y, test_x, test_y)
    print('\n')
    print("q2(e) Stochastic training: ")
    run_sgd(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
