from copy import deepcopy

from genRBF_source import cRBFkernel as f
import numpy as np

__author__ = "Łukasz Struski"


# read data potentially with missing values
def read_data(path, sep=','):
    return np.genfromtxt(path, delimiter=sep)


def whiten_matrix(covariance_matrix):
    EPS = 1e-10
    # eigenvalue decomposition of the covariance matrix
    d, E = np.linalg.eigh(covariance_matrix)
    d = np.divide(1., np.sqrt(d + EPS))
    W = np.einsum('ij, j, kj -> ik', E, d, E)
    return W


class RBFkernel(object):
    """
    Fast rbf kernel for missing data
    """

    def __init__(self, mean, covariance_matrix, data):
        np.atleast_2d(data)
        self.n_samples, self.n_features = data.shape
        self.data = data  # np.copy(data)
        self.mean = mean
        # self._info(self.mean)
        self._info(np.zeros(self.n_features))

        G = np.linalg.inv(covariance_matrix)
        reference = False
        if len(self.JJ) < self.n_features:
            self.G = G[np.ix_(self.JJ, self.JJ)]
            G_JJ_inv = np.linalg.inv(self.G)
            P_JJ = np.einsum('ij,jk->ik', G_JJ_inv, G[self.JJ, :])
            self.new_data = np.einsum('ij,kj->ki', P_JJ, self.data)
            self.mean = np.einsum('ij,j->i', P_JJ, self.mean)
            del G_JJ_inv, P_JJ

            self.Jx = deepcopy(self.J)
            f.updateFeatures(self.J, self.JJ)
        else:
            reference = True
            self.Jx = self.J
            self.G = np.copy(G)
            self.new_data = self.data
        self.Ps = f.change_data(self.data, self.new_data, self.mean, self.JJ, self.S, self.J, self.Jx, self.G,
                                reference=reference)
        self.Z = np.einsum('ij, kj->ki', G, self.data)
        del G

    def _info(self, fill):
        """
        This function collects informations about missing values and puts
        into missing coordinates from fill vector.
        :param data: dataset, numpy asrray, shape like (n_samples, n_features)
        :param fill: vector , numpy array, shape like (n_features)
        :return: list of indexes, list of missing sets, list of indexes of all points which have missing components
        """
        self.S, self.JJ, self.J, self.completeDataId = f.missingINFO(self.data, fill)

    def trainTestID(self, test_id):
        return f.trainTestID(test_id, self.S, self.completeDataId)

    def kernelTrain(self, gamma, train_id, S_train_new, completeDataId_train_new):
        return f.krenel_train(gamma, self.data[train_id, :], self.new_data[train_id, :], self.Z[train_id, :], self.G,
                              S_train_new, self.J, self.JJ, completeDataId_train_new, self.Ps)

    def kernelTest(self, gamma, test_id, train_id, S_test_new, S_train_new, completeDataId_test_new,
                   completeDataId_train_new):
        return f.krenel_test(test_id, train_id, gamma, self.data, self.new_data, self.Z, self.G, S_train_new,
                             S_test_new, self.J, self.JJ, completeDataId_train_new, completeDataId_test_new,
                             self.Ps)
