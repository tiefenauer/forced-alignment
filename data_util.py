import numpy as np


def create_labelled_data(corpus):
    train_set, dev_set, test_set = corpus.train_dev_test_split()

    N_train = len(train_set)
    X_train = np.zeros((N_train, N_train))
    Y_train = np.zeros((N_train, 1))

    N_dev = len(dev_set)
    X_dev = np.zeros((N_dev, N_train))
    Y_dev = np.zeros((N_dev, 1))

    N_test = len(test_set)
    X_test = np.zeros((N_test, N_train))
    Y_test = np.zeros((N_test, 1))

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test
