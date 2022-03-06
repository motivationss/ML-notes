import numpy as np
import math
import matplotlib.pyplot as plt


class Q1:
    def __init__(self, train_in, label_in, size_in):
        self.train = train_in
        self.label = label_in
        self.size = size_in
        self.weight = np.zeros((train_in.shape[1],1))

    def sigmoid(self, matrix_in):
        return 1 / (1 + np.exp(-np.dot(self.train, matrix_in)))

    def hessian(self, e_w_in):
        middle_term = self.sigmoid(e_w_in) * (1 - self.sigmoid(e_w_in)).reshape(1, self.size)
        dig_matrix = np.diag(np.diag(middle_term))
        return np.dot(np.dot(self.train.T, dig_matrix), self.train)

    def newton_method(self):
        for i in range(100):
            h = self.sigmoid(self.weight)
            e_w = np.dot(self.train.T, (h - self.label))
            self.weight -= np.dot(np.linalg.inv(self.hessian(self.weight)), e_w)
        print("The Weight Matrix is calculated as below: ")
        print(self.weight)

    def plot(self):
        self.newton_method()

        x1_label0 = list()
        x1_label1 = list()
        x2_label0 = list()
        x2_label1 = list()
        for i in range(self.size):
            if self.label[i] == 0:
                x1_label0.append(self.train[i, 1])
                x2_label0.append(self.train[i, 2])
            else:
                x1_label1.append(self.train[i, 1])
                x2_label1.append(self.train[i, 2])
        plt.scatter(x1_label0, x2_label0)
        plt.scatter(x1_label1, x2_label1)
        plt.legend(['0', '1'])
        x1_test = np.linspace(np.min(self.train[:,1]), np.max(self.train[:,1]), self.size).reshape(self.size, 1)
        x2_test = -(x1_test * self.weight[1] + self.weight[0]) / self.weight[2]
        plt.plot(x1_test, x2_test)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()


def main():
    # We format the data matrix so that each row is the feature for one sample.
    # The number of rows is the number of data samples.
    # The number of columns is the dimension of one data sample.
    train = np.load('q1x.npy')
    size = train.shape[0]
    label = np.load('q1y.npy')
    label = label.reshape(label.shape[0],1)
    # To consider intercept term, we append a column vector with all entries=1.
    # Then the coefficient correpsonding to this column is an intercept term.
    train = np.concatenate((np.ones((size, 1)), train), axis=1)
    solution = Q1(train, label, size)
    solution.plot()
    # print(solution.sigmoid(solution.weight))
    # print(1 / (1 + np.exp(-np.dot(train,solution.weight))))


if __name__ == "__main__":
    main()

