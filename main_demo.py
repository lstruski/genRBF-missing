import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun

__author__ = "Łukasz Struski"


# ____________________MAIN FUNCTION_____________________#

def main():
    if len(sys.argv) < 2:
        raise ValueError("Assuming write paths to dir which includes needed files")
    else:
        path_dir_data = sys.argv[1]

    # parameters for SVM
    C = 1
    gamma = 1.e-3

    precomputed_svm = SVC(C=C, kernel='precomputed')

    # read data
    m = np.genfromtxt(os.path.join(path_dir_data, 'mu.txt'), dtype=float, delimiter=',')
    cov = np.genfromtxt(os.path.join(path_dir_data, 'cov.txt'), dtype=float, delimiter=',')

    X_train = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    y_train = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    X_test = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    y_test = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')

    index_train = np.arange(X_train.shape[0])
    index_test = np.arange(X_test.shape[0]) + X_train.shape[0]
    X = np.concatenate((X_train, X_test), axis=0)
    del X_train, X_test

    index_train = index_train.astype(np.intc)
    index_test = index_test.astype(np.intc)

    # train
    rbf_ker = rbf.RBFkernel(m, cov, X)
    S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
                                                                                   rbf_ker.S)
    S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)

    train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)
    precomputed_svm.fit(train, y_train)

    # test
    S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
    test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
                              completeDataId_test_new, completeDataId_train_new)

    y_pred = precomputed_svm.predict(test)

    print("Accuracy classification score: {:.2f}".format(accuracy_score(y_test, y_pred)))


if __name__ == "__main__":
    main()

    pass
