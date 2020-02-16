__author__ = 'Ben Lengerich'

import numpy as np
from numpy import linalg


class Lasso:
    def __init__(self, lam=1e-4, lr=1., tol=np.finfo(float).eps, maxIter=500, random_restarts=5, lr_decay=0.999):
        self.lam = lam
        self.lr = lr
        self.initial_lr = lr
        self.tol = tol
        self.decay = lr_decay
        self.maxIter = maxIter
        self.random_restarts = random_restarts

    def fit(self, X, y, sample_weights=None, n_random_restarts=None, all_pos=False, verbose=False):
        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)
        self.N = X.shape[0]
        self.P = X.shape[1]
        self.D = y.shape[1]

        if sample_weights is None:
            sample_weights = np.ones((self.N))

        best_residue = np.inf
        self.best_beta = None
        if n_random_restarts is None:
            n_random_restarts = self.random_restarts
        for i in range(n_random_restarts):
            if verbose:
                print("Restart {:d}/{:d}".format(i+1, n_random_restarts), end='\t')
            self.beta = np.zeros((self.P, self.D))#np.random.normal(0, 1, (self.P, self.D))#np.zeros([shp[1], y.shape[1]])
            self.lr = self.initial_lr
            resi_prev = np.inf
            residue = self.cost(X, y, sample_weights)
            for step in range(self.maxIter):
                resi_prev = residue
                prev_beta = self.beta
                pg = self.proximal_gradient(X, y, sample_weights).reshape((self.P, self.D))
                self.beta -= pg*self.lr
                self.beta = self.proximal_proj(self.beta, all_pos)
                residue = self.cost(X, y, sample_weights)
                if residue > resi_prev:
                    if verbose:
                        print("Lasso converged after {:d} iterations. Residue: {:.3f}".format(step, residue), end='')
                    self.beta = prev_beta
                    break
                self.lr = self.decay * self.lr
            if verbose and step >= self.maxIter - 1:
                print("Lasso hit maximum iterations.        Residue: {:.3f}".format(residue), end='')
            if residue < best_residue:
                self.best_beta = self.beta
                best_residue = residue
                if verbose:
                    print("\t*")
            elif verbose:
                print("")

        self.beta = self.best_beta
        return self.beta


    def cost(self, X, y, sample_weights):
        return 0.5 * np.sum(sample_weights*(np.square(y - X.dot(self.beta)))) + self.lam*linalg.norm(self.beta, ord=1)


    def proximal_gradient(self, X, y, sample_weights=None):
        return 1.0/self.N * np.sum(np.array([
            sample_weights[i] * (X[i].dot(self.beta) - y[i]) * X[i] for i in range(self.N)]), axis=0)


    def proximal_proj(self, B, all_pos=False):
        t = self.lam * self.lr
        #result = np.maximum(0, B - t) - np.maximum(0, -B - t)   # ((maximum(abs(B), t) - t)*sign(B))
        if all_pos:
            result = (np.maximum(np.abs(B), t) - t)*np.sign(B)
            result[result < t] = 0.
        else:
            result = (np.maximum(np.abs(B), t) - t)*np.sign(B)
            result[np.abs(result) < t] = 0.

        return result

    def predict(self, X):
        y = np.dot(X, self.beta)

    def getBeta(self):
        #self.beta = self.beta.reshape(self.beta.shape[0])
        return self.beta

    def setLambda(self, lam):
        self.lam = lam

    def setLearningRate(self, lr):
        self.lr = lr

    def setMaxIter(self, a):
        self.maxIter = a

    def setTol(self, t):
        self.tol = t