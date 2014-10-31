import numpy as np


class LeeSeungNonnegativeMatrixFactorization:
    """
    "Abstract" non-negative matrix factorization as in the paper from Lee & Seung:
    Algorithms for non-negative matrix factorization (NIPS 2001)
    We follow here the same formalism:
        - the n by m data matrix is denoted V.
        - the factor matrices are W (n by r) and H (r by m)
    """
    def __init__(self, n, m, r, random_seed=0):
        self.n, self.m, self.r = n, m, r
        np.random.seed(random_seed)
        self.W = np.random.random((n, r))
        self.H = np.random.random((r, m))


class EuclideanLeeSeungNonnegativeMatrixFactorization(LeeSeungNonnegativeMatrixFactorization):
    """
    Implementation of the update rules for Mean Squared Error loss.
    """
    def __init__(self, n, m, r):
        LeeSeungNonnegativeMatrixFactorization.__init__(self, n, m, r)

    def update_factors(self, V):
        self.H *= np.dot(np.transpose(self.W), V) / np.dot(np.dot(np.transpose(self.W), self.W), self.H)
        self.W *= np.dot(V, np.transpose(self.H)) / np.dot(self.W, np.dot(self.H, np.transpose(self.H)))

    def compute_loss(self, V):
        return np.linalg.norm(V - np.dot(self.W, self.H)) ** 2


class DivergenceLeeSeungNonnegativeMatrixFactorization(LeeSeungNonnegativeMatrixFactorization):
    """
    Implementation of the update rules for divergence loss (linked to Kullback-Leibler divergence).
    """
    def __init__(self, n, m, r):
        LeeSeungNonnegativeMatrixFactorization.__init__(self, n, m, r)

    def update_factors(self, V):
        # The [:, None] is a trick to force correct broadcasting for np.divide
        self.H *= np.dot(np.transpose(self.W), V / np.dot(self.W, self.H)) / np.sum(self.W, axis=0)[:, None]
        self.W *= np.dot(V / np.dot(self.W, self.H), np.transpose(self.H)) / np.sum(self.H, axis=1)

    def compute_loss(self, V):
        # Compute WH only once.
        WH = np.dot(self.W, self.H)
        return np.sum(V * np.log(1e-10 + V / WH) - V + WH)


def get_model(nmf_type, n, m, r):
    if nmf_type == u"euclidean":
        return EuclideanLeeSeungNonnegativeMatrixFactorization(n, m, r)
    elif nmf_type == u"divergence":
        return DivergenceLeeSeungNonnegativeMatrixFactorization(n, m, r)
    else:
        raise ValueError(u"Invalid NMF type: {0}".format(nmf_type))