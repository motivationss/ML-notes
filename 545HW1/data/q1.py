import numpy as np
import matplotlib.pyplot as plt

q1xTrain = np.load("q1xTrain.npy")
q1yTrain = np.load("q1yTrain.npy")
q1xTest = np.load("q1xTest.npy")
q1yTest = np.load("q1yTest.npy")


class GradientDescent:
    def __init__(self, train_in, label_in):
        self.w0 = 0
        self.w1 = 0
        self.lr = 0.01
        self.train = train_in
        self.label = label_in
        self.size = len(train_in)

    def train_batch_gd(self):
        epoch = 0
        error = 10
        while error > 0.2:
            epoch = epoch + 1
            gd_w0 = sum([self.w0 + self.w1 * self.train[i] - self.label[i] for i in range(self.size)])
            gd_w1 = sum([(self.w0 + self.w1 * self.train[i] - self.label[i]) * self.train[i] for i in range(self.size)])
            self.w0 = self.w0 - self.lr * gd_w0
            self.w1 = self.w1 - self.lr * gd_w1

            error = self.mean_squared_error()
        print("---batch gradient descent stats---")
        print("current w0: ", self.w0)
        print("current w1: ", self.w1)
        print("Learning rates: ", self.lr)
        print("Number of Epochs: ", epoch)

    def train_stochastic_gd(self):
        epoch = 0
        error = 10
        while error > 0.2:
            epoch += 1
            for index in range(self.size):
                gd_w0 = (self.w0 + self.w1 * self.train[index] - self.label[index])
                gd_w1 = (self.w0 + self.w1 * self.train[index] - self.label[index]) * self.train[index]
                self.w0 = self.w0 - self.lr * gd_w0
                self.w1 = self.w1 - self.lr * gd_w1
                error = self.mean_squared_error()

                if error <= 0.2:
                    break

        print("---stochastic gradient descent stats---")
        print("current w0: ", self.w0)
        print("current w1: ", self.w1)
        print("Learning rates: ", self.lr)
        print("Number of Epochs: ", epoch)

    def mean_squared_error(self):
        return sum([(self.w0 + self.w1 * self.train[i] - self.label[i]) ** 2 for i in range(self.size)]) / self.size


class ClosedForm:
    def __init__(self, train_data_in, train_label_in, test_data_in, test_label_in):
        self.lr = 0.01
        self.size = len(train_data_in)
        self.train = np.reshape(train_data_in, (self.size, 1))
        self.label = train_label_in
        self.test_data = np.reshape(test_data_in, (self.size, 1))
        self.test_label = test_label_in

    def generate_phi_matrix(self, m_degree, data):
        matrix = np.ones((self.size, 1), dtype=float)
        for index in range(1, m_degree+1):
            concat_train = data ** index
            matrix = np.concatenate((matrix, concat_train), axis=1)
        return matrix

    def generate_w_matrix(self, phi_matrix):
        return np.dot(np.linalg.inv(np.dot(phi_matrix.T, phi_matrix)), np.dot(phi_matrix.T, self.label))

    def closed_rms(self, w_matrix, phi_matrix, label):

        objective = (1/2) * sum([(np.dot(w_matrix.T, phi_matrix[i]) - label[i]) ** 2 for i in range(self.size)])
        return np.sqrt(2 * objective / self.size)

    def figure_1b(self):
        degree_list = []
        error_train_data = []
        error_test_data = []
        for i in range(10):
            degree_list.append(i)

            train_phi_matrices = self.generate_phi_matrix(m_degree=i, data=self.train)
            w_matrices = self.generate_w_matrix(phi_matrix=train_phi_matrices)
            error_train_data.append(self.closed_rms(w_matrices, train_phi_matrices, self.label))

            test_phi_matrices = self.generate_phi_matrix(m_degree=i,data=self.test_data)
            error_test_data.append(self.closed_rms(w_matrices, test_phi_matrices, self.test_label))

        plt.plot(degree_list, error_train_data)
        plt.plot(degree_list, error_test_data)
        plt.legend(["Training", "Test"])
        plt.xlabel("M")
        plt.ylabel("E_RMS")
        plt.show()

    def regularized_weight(self, phi_matrix, lamda):
        regu = lamda  * np.identity(np.dot(phi_matrix.T, phi_matrix).shape[0])
        return np.dot(np.linalg.inv(regu + np.dot(phi_matrix.T, phi_matrix)), np.dot(phi_matrix.T, self.label))

    def figure_1c(self):
        lamda_list = []
        error_train_data = []
        error_test_data = []
        lamda_list.append(10 ** (-14))
        degree = 9
        for i in range(-8, 1):
            lamda_list.append(10 ** i)

        train_phi_matrices = self.generate_phi_matrix(m_degree=degree, data=self.train)
        test_phi_matrices = self.generate_phi_matrix(m_degree=degree, data=self.test_data)

        for lm in lamda_list:
            w_matrices = self.regularized_weight(phi_matrix=train_phi_matrices, lamda=lm)
            error_train_data.append(self.closed_rms(w_matrices, train_phi_matrices, self.label))
            error_test_data.append(self.closed_rms(w_matrices, test_phi_matrices, self.test_label))

        plt.plot(np.log(lamda_list), error_train_data)
        plt.plot(np.log(lamda_list), error_test_data)
        plt.legend(["Training", "Test"])
        plt.xlabel("ln lambda")
        plt.ylabel("E_RMS")
        plt.show()


if __name__ == '__main__':
    batch_model = GradientDescent(q1xTrain, q1yTrain)
    batch_model.train_batch_gd()
    sgd_model = GradientDescent(q1xTrain, q1yTrain)
    sgd_model.train_stochastic_gd()
    closed_model = ClosedForm(q1xTrain, q1yTrain, q1xTest, q1yTest)
    closed_model.figure_1b()
    closed_model.figure_1c()
