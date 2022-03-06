import numpy as np
import matplotlib.pyplot as plt


def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
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


def nb_train(matrix, category):
    # Implement your algorithm and return
    train_spam, train_nonspam = matrix[category == 1], matrix[category == 0]
    num_spam, num_nonspam = np.sum(train_spam), np.sum(train_nonspam)
    word_spam, word_nonspam = np.sum(train_spam, axis=0), np.sum(train_nonspam, axis=0)

    state = {}
    bernouli_spams = {0: 1 - np.sum(category) / category.shape[0], 1: np.sum(category) / category.shape[0]}
    condition_spams = {(i, 1): (j + 1) / (num_spam + len(matrix[0])) for i, j in enumerate(word_spam)}
    condition_nonspams = {(i, 0): (j + 1) / (num_nonspam + len(matrix[0])) for i, j in enumerate(word_nonspam)}
    state.update(bernouli_spams)
    state.update(condition_spams)
    state.update(condition_nonspams)
    return state


def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    # for item in matrix:
    output = np.zeros(800)
    curr_row = 0
    for item in matrix:
        p_nonspams = [sum([j * np.log(state[i, 0]) for i, j in enumerate(item)])]
        p_spams = [sum([j * np.log(state[i, 1]) for i, j in enumerate(item)])]
        if p_spams > p_nonspams:
            output[curr_row] = 1
        curr_row += 1
    return output


def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))


def part_b(state, tokens, num_words):
    all_words = [(np.log(state[i, 1]) - np.log(state[i, 0]), i) for i in range(num_words)]
    all_words.sort(reverse=True)
    for num, (junk, token_index) in enumerate(all_words):
        print("Token ", num, ": ", tokens[token_index])
        if num == 4:
            break

def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')
    # print(dataMatrix_train[category_train == 1])
    # Train
    state = nb_train(dataMatrix_train, category_train)
    # # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    print("Part(a): ")
    evaluate(prediction, category_test)
    print("")
    # part B
    print("Part(b): ")
    part_b(state, tokenlist, len(dataMatrix_train[0]))
    print("")
    # part C
    diff_size = [50, 100, 200, 400, 800, 1400]
    error = list()
    print("Part(c): ")
    for i in diff_size:
        dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN.' + str(i))
        state = nb_train(dataMatrix_train, category_train)
        prediction = nb_test(dataMatrix_test, state)
        this_error = (prediction != category_test).sum() * 1. / len(prediction)
        print('Error: {:2.4f}%'.format(100 * this_error))
        error.append(this_error)
    plt.plot(diff_size, error)
    plt.xlabel("size")
    plt.ylabel("error")
    plt.show()


if __name__ == "__main__":
    main()
