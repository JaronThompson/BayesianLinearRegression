'''
LR.py by Jaron Thompson

This script includes the BLR class for implementing Bayesian linear regression.
Precision hyper-parameters are updated based on the training data by maximizing
the evidence function using the Expecation-Maximization algorithm.

Also included is a GLR class for general linear regression.

'''
import numpy as np

### Bayesian Linear Regression model ###

class BLR:

    def __init__(self, X, Y, alpha=1e-3, beta=1.0, tol=1e-3):
        self.X = X
        self.Y = Y.ravel()

        # number of samples and basis functions
        self.n_samples, self.n_basis = X.shape

        # hyper parameters
        self.alpha = alpha*np.ones(self.n_basis)
        self.beta  = beta

        # parameters of hyper-prior
        self.a = 1e-4

        # convergence tolerance
        self.tol = tol

    def fit(self):
        convergence = np.inf
        prev_evidence = 0
        while convergence > self.tol:
            # E step
            A = self.alpha*np.eye(self.n_basis) + self.beta*self.X.T@self.X
            self.Ainv = np.linalg.inv(A)
            self.Ainv = 1/2*(self.Ainv + self.Ainv.T)
            self.mu = self.beta*self.Ainv@self.X.T@self.Y
            Y_pred = self.X@self.mu
            self.SSE = np.sum((Y_pred - self.Y)**2)

            # M step
            gamma = np.sum(1. - self.alpha*np.diag(self.Ainv))
            # gamma = self.n_basis - self.alpha*np.trace(self.Ainv)
            self.alpha = np.ones(self.n_basis) * (1. + self.a) / (np.dot(self.mu,self.mu) + np.trace(self.Ainv) + self.a)
            # self.alpha = (self.n_basis + self.a) / (np.dot(self.mu, self.mu) + np.trace(self.Ainv) + self.a)
            self.beta  = (self.n_samples + self.a) / (self.SSE + gamma/self.beta + self.a)

            # evaluate convergence
            current_evidence = self.evidence()
            print("Evidence: {:.3f}".format(current_evidence))
            convergence = np.abs(prev_evidence - current_evidence) / np.max([1,np.abs(prev_evidence)])
            prev_evidence = current_evidence

    def evidence(self):
        ev = np.sum(np.log(np.linalg.eigvalsh(self.Ainv))) + \
             np.sum(np.log(self.alpha)) + \
             self.n_samples*np.log(self.beta) - self.beta*self.SSE - \
             np.dot(self.alpha*self.mu, self.mu)
        return ev/2.

    def predict(self, X):
        y_pred = X@self.mu
        y_var  = 1/self.beta + np.einsum('ni,ij,nj->n', X, self.Ainv, X)
        return y_pred, np.sqrt(y_var)

class GLR:

    def __init__(self, X, Y, lam=0.0):
        self.X = X
        self.Y = Y
        self.lam = 0.0
        self.NS, self.NF = X.shape

    def W_ML(self, lam):
        # find maximum likelihood parameter estimates
        pseudo_inverse = np.dot(np.linalg.inv(lam*np.eye(self.NF) + np.dot(self.X.T, self.X)), self.X.T)
        W = np.dot(pseudo_inverse, self.Y)
        return W

    def predict(self, x):
        return np.dot(self.W_ML(self.lam).T, x.T)
