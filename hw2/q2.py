# work with Zhiyuan Sun for this Q

import numpy as np
import matplotlib.pyplot as plt

# Load data
q2_data = np.load('q2_data.npz')


def compute_grad(x_train, y_train, target_y, denominator_in):
    return np.sum(x_train * ((y_train == target_y) - denominator_in[:, target_y - 1] /
                             np.sum(denominator_in, axis=1)).reshape(-1, 1), axis=0)


if __name__ == "__main__":
    lr = 0.0005
    train_x, train_y = q2_data["q2x_train"], q2_data["q2y_train"].astype(int).flatten()
    test_x, test_y = q2_data["q2x_test"], q2_data["q2y_test"].astype(int).flatten()
    num_class = len(np.unique(train_y))
    weight = np.zeros((num_class, train_x.shape[1])) # n by 3 matrix, row = num of data
    denominator = np.exp(train_x @ weight.T)
    lens = len(denominator)
    a = denominator[np.arange(lens), (train_y - 1).flatten()] # get (a,b) from first and second consecutively
    b = np.sum(denominator, axis=1)  # sum by rows
    l_obj = np.sum(a / b)

    l_obj_last = -1000000
    predict = 1
    while np.linalg.norm(l_obj - l_obj_last) / np.linalg.norm(l_obj_last) > 0.0001:
        grad_1 = compute_grad(train_x, train_y, 1, denominator)
        weight[0, :] = weight[0, :] + lr * grad_1
        grad_2 = compute_grad(train_x, train_y, 2, denominator)
        weight[1, :] = weight[1, :] + lr * grad_2
        l_obj_last = l_obj
        denominator = np.exp(train_x @ weight.T)
        a = denominator[np.arange(lens), (train_y - 1).flatten()]
        b = np.sum(denominator, axis=1)
        l_obj = np.sum(a / b)
        # test_x is 50 x 4 and w.T is 4 x 3 so 50 by 3 take the max indices of each row
        predict = np.argmax(test_x @ weight.T, axis=1) + 1
    print("accuracy: ", np.sum(test_y == predict) / len(test_y))
