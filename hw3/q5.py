# EECS 545 HW3 Q5

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

np.random.seed(545)

def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        return matrix, tokens, np.array(Y)


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error


def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    # Train
    classifier = LinearSVC(max_iter=20000)
    classifier.fit(dataMatrix_train, category_train)
    # Test and evluate
    prediction = classifier.predict(dataMatrix_test)
    print("Part a: ")
    evaluate(prediction, category_test)

    # part b
    print("\nPart b: ")
    num_trains = [50, 100, 200, 400, 800, 1400]
    error_list = list()
    for train_size in num_trains:
        dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN.' + str(train_size))
        classifier = LinearSVC(max_iter=20000)
        classifier.fit(dataMatrix_train, category_train)
        prediction = classifier.predict(dataMatrix_test)
        support_vector = classifier.decision_function(dataMatrix_train)
        num_support_vectors = 0
        for item in support_vector:
            if abs(item) <= 1:
                num_support_vectors +=1
        print("Data size: ", str(train_size))
        print("Number of support vectors: ", str(num_support_vectors))
        error_list.append(evaluate(prediction, category_test))


    plt.plot(num_trains, error_list)
    plt.xlabel("train size")
    plt.ylabel("error")
    plt.show()


if __name__ == '__main__':
    main()
