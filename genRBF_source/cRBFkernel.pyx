#!python
#cython: boundscheck=False

import numpy as np

from libcpp.map cimport map

__author__ = "Łukasz Struski"


def missingINFO(double[:,:] data, double[:] fill):
    cdef:
        int i = 0, j = 0, k = 0, size_ = 0
        list S = [], J = [], x = [], JJ = [], completeDataId = []
        set JJ_ = set()
        bint b1, b2

    for i in range(data.shape[0]):
        x = []
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                x.append(j)
        size_ = len(x)
        if size_:
            for j in range(size_):
                k = x[j]
                data[i, k] = fill[k]
            if len(J):
                b1 = False
                for j in range(len(J)):
                    if size_ == len(J[j]):
                        b2 = True
                        for k in range(size_):
                            if x[k] != J[j][k]:
                                b2 = False
                                break
                        if b2:
                            b1 = True
                            break
                if b1:
                    S[j].append(i)
                else:
                    J.append(x)
                    S.append([i])
                    JJ_.update(x)
            else:
                J.append(x)
                S.append([i])
                JJ_.update(x)
        else:
            completeDataId.append(i)

    JJ = sorted(JJ_)
    del JJ_
    return (S, JJ, J, completeDataId)


def change_data(X, new_X, double[:] new_m, list JJ, list S, list J, list Jx, G, bint reference=False):
    cdef int i = 0
    cdef double[:,:] temp

    cdef int _size = len(S)

    Ps = np.empty((_size,), dtype=object)

    for i in range(_size):
        ps = np.linalg.inv(G[np.ix_(J[i], J[i])])
        ps = np.einsum('ij,jk->ik', ps, G[np.ix_(J[i], JJ)])

        Ps[i] = ps

        temp = new_m - new_X[S[i], :]
        temp = np.einsum('ij,kj->ki', ps, temp)

        X[np.ix_(S[i], Jx[i])] += temp
        if not reference:
            new_X[np.ix_(S[i], J[i])] += temp
    return Ps

def krenel_train(double gamma, X, new_X, Z, G, list S, list J, list JJ, list completeDataId, Ps):
    cdef:
        int i = 0, j = 0, ii = 0, jj = 0, kk = 0, l = 0, ll = 0, id_ = 0, size_ = 0, size2 = 0
        double [:, :] X_view = X, Z_view = Z
        double scalar = 0., z = 0.
        int n_samples = X.shape[0]
        int n_features = X.shape[1]

    gramRBF = np.empty((n_samples, n_samples), dtype=float)
    cdef double [:, :] gramRBF_view = gramRBF

    # case I (diagonal)
    for i in range(n_samples):
        gramRBF_view[i, i] = 1.

    # case II (no missing)
    size_ = len(completeDataId)
    for i in range(size_):
        kk = completeDataId[i]
        for j in range(i + 1, size_):
            jj = completeDataId[j]
            scalar = 0.
            for ii in range(n_features):
                scalar += (X_view[kk, ii] - X_view[jj, ii]) * (Z_view[kk, ii] - Z_view[jj, ii])
            gramRBF_view[kk, jj] = np.exp(-gamma * scalar)
            gramRBF_view[jj, kk] = gramRBF_view[kk, jj]

    # case III (one missing)
    for i in range(size_):
        for id_ in range(len(S)):
            Gs = G[np.ix_(J[id_], J[id_])]
            z = len(J[id_])
            z = np.power(1 + 4 * gamma, z / 4.0) / np.power(1 + 2 * gamma, z / 2.0)

            ps = Ps[id_]

            temp_x = new_X[completeDataId[i], :] - new_X[S[id_], :]
            p = np.einsum('ij,kj->ik', ps, temp_x)
            r = np.einsum('ji,jk,ki->i', p, Gs, p)
            temp_x = X[completeDataId[i], :] - X[S[id_], :]
            temp_z = Z[completeDataId[i], :] - Z[S[id_], :]
            w = np.einsum('ij,ij->i', temp_x, temp_z) - (2 * gamma) / (1 + 2 * gamma) * r
            gramRBF[completeDataId[i], S[id_]] = z * np.exp(-gamma * w)
            gramRBF[S[id_], completeDataId[i]] = gramRBF[completeDataId[i], S[id_]]

    # case IV (two missing)
    size_ = len(S)
    for i in range(size_):
        Gs = G[np.ix_(J[i], J[i])]
        z = 1

        ps = Ps[i]

        for j in range(i, size_):
            if i == j:
                size2 = len(S[i])
                for ii in range(size2):
                    l = S[i][ii]
                    for jj in range(ii + 1, size2):
                        ll = S[i][jj]
                        temp_x = new_X[l, :] - new_X[ll, :]
                        p = np.einsum('ij,j->i', ps, temp_x)
                        r = np.einsum('i,ij,j', p, Gs, p)
                        scalar = 0.
                        for kk in range(n_features):
                            scalar += (X_view[l, kk] - X_view[ll, kk]) * (Z_view[l, kk] - Z_view[ll, kk])
                        w = scalar - 4 * gamma * r / (1 + 4 * gamma)
                        gramRBF_view[l, ll] = z * np.exp(-gamma * w)
                        gramRBF_view[ll, l] = gramRBF_view[l, ll]
            else:
                ps_j = Ps[j]

                J_ij = set(J[i])
                J_ij.update(J[j])
                J_ij = sorted(J_ij)
                size2 = len(JJ)
                Q = np.zeros((size2, size2))
                Q[J[i], :] += ps
                Q[J[j], :] += ps_j
                I = np.identity(size2)
                Q = I[np.ix_(J_ij, J_ij)] + 2 * gamma * Q[np.ix_(J_ij, J_ij)]
                Q = np.linalg.inv(Q)
                R = Q - I[np.ix_(J_ij, J_ij)] # I_{J_i \cup J_j}
                R = np.einsum('ij,jk->ik', G[np.ix_(J_ij, J_ij)], R)
                del I
                z = np.power(1 + 4 * gamma, (len(J[i]) + len(J[j])) / 4.) * np.sqrt(np.linalg.det(Q))
                del Q

                ps_ij = np.linalg.inv(G[np.ix_(J_ij, J_ij)])
                ps_ij = np.einsum('ij,jk->ik', ps_ij, G[np.ix_(J_ij, JJ)])

                for ii in range(len(S[i])):
                    temp_x = new_X[S[i][ii], :] - new_X[S[j], :]
                    v = np.einsum('ij,kj->ki', ps_ij, temp_x)
                    temp_x = X[S[i][ii], :] - X[S[j], :]
                    temp_z = Z[S[i][ii], :] - Z[S[j], :]
                    w = np.einsum('ij,ij->i', temp_x, temp_z) - np.einsum('ij,jk,ik->i', v, R, v)
                    gramRBF[S[i][ii], S[j]] = z * np.exp(-gamma * w)
                    gramRBF[S[j], S[i][ii]] = gramRBF[S[i][ii], S[j]]
    return gramRBF


def krenel_test(indexes_test, indexes_train, gamma, X, new_X, Z, G, S_train, S_test, J, JJ,
                completeDataId_train, completeDataId_test, Ps):
    cdef:
        int i = 0, j = 0, ii = 0, jj = 0, size1 = 0, size2 = 0, kk = 0, id_ = 0, size_ = 0, l = 0, ll =0
        double [:, :] X_view = X, Z_view = Z
        double scalar = 0., z = 0.
        int n_samples = X.shape[0]
        int n_features = X.shape[1]

    X_train = X[indexes_train, :]
    X_test = X[indexes_test, :]
    new_X_train = new_X[indexes_train, :]
    new_X_test = new_X[indexes_test, :]
    Z_train = Z[indexes_train, :]
    Z_test = Z[indexes_test, :]
    cdef:
        double [:, :] X_train_view = X_train, X_test_view = X_test
        double [:, :]  new_X_train_view = new_X_train, new_X_test_view = new_X_test
        double [:, :]  Z_train_view = Z_train, Z_test_view = Z_test

    gramRBF = np.empty((len(indexes_test), len(indexes_train)), dtype=float)
    cdef double [:, :] gramRBF_view = gramRBF

    # case I (no missing)
    size1 = len(completeDataId_test)
    size2 = len(completeDataId_train)
    for i in range(size1):
        kk = completeDataId_test[i]
        for j in range(size2):
            jj = completeDataId_train[j]
            scalar = 0.
            for ii in range(n_features):
                scalar += (X_test_view[kk, ii] - X_train_view[jj, ii]) * (Z_test_view[kk, ii] - Z_train_view[jj, ii])
            gramRBF_view[kk, jj] = np.exp(-gamma * scalar)

    # case II (one missing)
    # no missing from test and missing from train
    for i in range(size1):
        for id_ in range(len(S_train)):
            Gs = G[np.ix_(J[id_], J[id_])]
            z = len(J[id_])
            z = (1 + 4 * gamma) ** (z / 4.0) / (1 + 2 * gamma) ** (z / 2.0)

            ps = Ps[id_]

            temp_x = new_X_test[completeDataId_test[i], :] - new_X_train[S_train[id_], :]
            p = np.einsum('ij,kj->ik', ps, temp_x)
            r = np.einsum('ji,jk,ki->i', p, Gs, p)
            temp_x = X_test[completeDataId_test[i], :] - X_train[S_train[id_], :]
            temp_z = Z_test[completeDataId_test[i], :] - Z_train[S_train[id_], :]
            w = np.einsum('ij,ij->i', temp_x, temp_z) - (2 * gamma) / (1 + 2 * gamma) * r
            gramRBF[completeDataId_test[i], S_train[id_]] = z * np.exp(-gamma * w)

    # missing from test and no missing from train
    for i in range(size2):
        for id_ in range(len(S_test)):
            Gs = G[np.ix_(J[id_], J[id_])]
            z = len(J[id_])
            z = (1 + 4 * gamma) ** (z / 4.0) / (1 + 2 * gamma) ** (z / 2.0)

            ps = Ps[id_]

            temp_x = new_X_train[completeDataId_train[i], :] - new_X_test[S_test[id_], :]
            p = np.einsum('ij,kj->ik', ps, temp_x)
            r = np.einsum('ji,jk,ki->i', p, Gs, p)
            temp_x = X_train[completeDataId_train[i], :] - X_test[S_test[id_], :]
            temp_z = Z_train[completeDataId_train[i], :] - Z_test[S_test[id_], :]
            w = np.einsum('ij,ij->i', temp_x, temp_z) - (2 * gamma) / (1 + 2 * gamma) * r
            gramRBF[S_test[id_], completeDataId_train[i]] = z * np.exp(-gamma * w)

    # case III (two missing)
    size_ = len(S_test) # S_test and S_train must be the same number of list
    for i in range(size_):
        Gs = G[np.ix_(J[i], J[i])]
        z = 1

        ps = Ps[i]

        for j in range(size_):
            if i == j:
                size1 = len(S_test[i])
                for ii in range(size1):
                    temp_x = new_X_test[S_test[i][ii], :] - new_X_train[S_train[i], :]
                    p = np.einsum('ij,kj->ki', ps, temp_x)
                    r = np.einsum('ij,jk,ik->i', p, Gs, p)
                    temp_x = X_test[S_test[i][ii], :] - X_train[S_train[i], :]
                    temp_z = Z_test[S_test[i][ii], :] - Z_train[S_train[i], :]
                    w = np.einsum('ij,ij->i', temp_x, temp_z) - 4 * gamma * r / (1 + 4 * gamma)
                    gramRBF[S_test[i][ii], S_train[i]] = z * np.exp(-gamma * w)
            else:
                ps_j = Ps[j]

                J_ij = set(J[i])
                J_ij.update(J[j])
                J_ij = sorted(J_ij)
                size2 = len(JJ)
                Q = np.zeros((size2, size2))
                Q[J[i], :] += ps
                Q[J[j], :] += ps_j
                I = np.identity(size2)
                Q = I[np.ix_(J_ij, J_ij)] + 2 * gamma * Q[np.ix_(J_ij, J_ij)]
                Q = np.linalg.inv(Q)
                R = Q - I[np.ix_(J_ij, J_ij)]  # I_{J_i \cup J_j}
                R = np.einsum('ij,jk->ik', G[np.ix_(J_ij, J_ij)], R)
                del I
                z = (1 + 4 * gamma) ** ((len(J[i]) + len(J[j])) / 4.) * np.sqrt(np.linalg.det(Q))
                del Q

                ps_ij = np.linalg.inv(G[np.ix_(J_ij, J_ij)])
                ps_ij = np.einsum('ij,jk->ik', ps_ij, G[np.ix_(J_ij, JJ)])

                for ii in range(len(S_test[i])):
                    temp_x = new_X_test[S_test[i][ii], :] - new_X_train[S_train[j], :]
                    v = np.einsum('ij,kj->ki', ps_ij, temp_x)
                    temp_x = X_test[S_test[i][ii], :] - X_train[S_train[j], :]
                    temp_z = Z_test[S_test[i][ii], :] - Z_train[S_train[j], :]
                    w = np.einsum('ij,ij->i', temp_x, temp_z) - np.einsum('ij,jk,ik->i', v, R, v)
                    gramRBF[S_test[i][ii], S_train[j]] = z * np.exp(-gamma * w)

    return gramRBF


def trainTestID(int[:] test_id, list S, list completeDataId):
    cdef:
        int i = 0, j = 0, k = 0, l = 0, size_ = len(S), size1 = 0
        bint check = True
        list S_test = [[] for i in range(size_)]
        list S_train = []
        list completeDataId_test = []
        list completeDataId_train = []

    # compute S_test and completeDataId_test
    for i in range(len(test_id)):
        check = True
        for j in range(size_):
            for k in range(len(S[j])):
                if test_id[i] == S[j][k]:
                    S_test[j].append(test_id[i])
                    check = False
                    break
            if not check:
                break
        if check:
            completeDataId_test.append(test_id[i])

    # compute S_train
    for i in range(size_):
        S_train.append([])
        l = 0
        size1 = len(S[i])
        for j in range(len(S_test[i])):
            for k in range(l, size1):
                if S[i][k] < S_test[i][j]:
                    S_train[i].append(S[i][k])
                else:
                    l = k + 1
                    break
        for j in range(l, size1):
            S_train[i].append(S[i][j])

    # compute completeDataId
    size1 = len(completeDataId)
    l = 0
    for i in range(len(completeDataId_test)):
        for j in range(l, size1):
            if completeDataId[j] < completeDataId_test[i]:
                completeDataId_train.append(completeDataId[j])
            else:
                l = j + 1
                break
    for i in range(l, size1):
        completeDataId_train.append(completeDataId[i])
    return (S_train, S_test, completeDataId_train, completeDataId_test)


def trainTestID_1(int[:] test_id, int[:] train_id, list S):
    cdef:
        int size_ = len(S)
        list S_test = [[] for _ in range(size_)]
        list S_train = [[] for _ in range(size_)]
        list completeDataId_test = []
        list completeDataId_train = []

        int size1 = len(test_id)
        int size2 = len(train_id)
        int size3 = max(size1, size2)
        int i, j
        bint check1, check2

    for i in range(size3):
        if i < size1:
            check1 = True
        else:
            check1 = False
        if i < size2:
            check2 = True
        else:
            check2 = False
        for j in range(size_):
            if check1 and test_id[i] in S[j]:
                S_test[j].append(test_id[i])
                check1 = False
            elif check2 and train_id[i] in S[j]:
                S_train[j].append(train_id[i])
                check2 = False
            if not (check1 or check2):
                break
        if check1:
            completeDataId_test.append(test_id[i])
        if check2:
            completeDataId_train.append(train_id[i])

    return (S_train, S_test, completeDataId_train, completeDataId_test)

def updateSamples(train_id, S_train, completeDataId_train):
    # starting index from 0
    cdef:
        int size_ = len(S_train)
        list S_train_new, completeDataId_train_new

    my_dict = {train_id[i]: i for i in range(len(train_id))}
    cdef map[int, int] m = my_dict

    S_train_new = [[] for _ in range(size_)]
    for i in range(size_):
        for j in range(len(S_train[i])):
            S_train_new[i].append(m[S_train[i][j]])

    completeDataId_train_new = []
    for i in range(len(completeDataId_train)):
        completeDataId_train_new.append(m[completeDataId_train[i]])

    return (S_train_new, completeDataId_train_new)


def updateFeatures(list J, list JJ):
    cdef:
        int size_ = 0, i = 0, j = 0

    size_ = len(JJ)
    my_dict = {JJ[i]: i for i in range(size_)}
    cdef map[int, int] m = my_dict
    for i in range(len(J)):
        for j in range(len(J[i])):
            k = J[i][j]
            J[i][j] = m[k]

    for i in range(size_):
        JJ[i] = m[JJ[i]]
    return (J, JJ)
