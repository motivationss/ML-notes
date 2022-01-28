import numpy as np
import matplotlib.pyplot as plt

train_in = np.load("q2x.npy")
label_in = np.load("q2y.npy")


def unweighted_line(train, label):
    train = np.reshape(train, (len(train), 1))
    label = np.reshape(label, (len(label), 1))
    one = np.ones((len(train),1))
    phi_matrix = np.concatenate((one, train), axis=1)
    weights = np.dot(np.dot(np.linalg.inv(np.dot(phi_matrix.T, phi_matrix)),phi_matrix.T),label)
    w0 = weights[0]
    w1 = weights[1]
    test = np.linspace(-4.9779585, 11.85335, 50) # largest and lowest value
    predict = test * w1 + w0
    print("----Q2(d)(i)-----")
    print("Slope (w1): ", w1[0])
    print("Intercept (w0): ", w0[0])
    plt.xlabel("train")
    plt.ylabel("label")
    plt.plot(train, label, '.')
    plt.plot(test, predict, '')
    plt.legend(["data point", "fitted line"])
    plt.show()


def r_weights(train, point, bandwidth):
    r_weight = np.zeros((len(train), len(train)))
    for index in range(len(train)):
        r_weight[index][index] = np.exp(- ( (point - train[index]) ** 2 ) / (2 * (bandwidth**2)))
    return r_weight


def weighted_line(train, label, bandwidth=0.8):
    train = np.reshape(train, (len(train), 1))
    label = np.reshape(label, (len(label), 1))
    one = np.ones((len(train), 1))
    phi_matrix = np.concatenate((one, train), axis=1)
    test = np.linspace(-4.9779585, 11.85335, 50)
    predict = []
    for x in test:
        r_weight = r_weights(train, x, bandwidth)
        weights = np.dot(np.linalg.inv(np.dot(np.dot(phi_matrix.T, r_weight), phi_matrix)), phi_matrix.T)
        weights = np.dot(np.dot(weights, r_weight),label)
        w0 = weights[0][0]
        w1 = weights[1][0]
        predict.append(w1 * x + w0)
    plt.plot(train, label, '.')
    plt.plot(test, predict, '')
    plt.title("bandwidth: " + str(bandwidth))
    plt.xlabel("train")
    plt.ylabel("label")
    plt.show()


# using for the third question of question 2 D
def weighted_many_bandwidth(train, label):
    bandwidth_list = [0.1, 0.3, 2, 10]
    print("below are answers for Q2 d(iii):")
    for i in bandwidth_list:
        weighted_line(train, label, i)


if __name__ == '__main__':
    unweighted_line(train_in, label_in)
    weighted_line(train_in, label_in)
    weighted_many_bandwidth(train_in, label_in)