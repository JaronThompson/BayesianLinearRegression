import numpy as np
from scipy.optimize import fsolve

class BLR:

    def __init__(self, X, Y, a=.05, b=.05, max_evidence=False):

        self.X = X
        self.Y = Y
        self.NS, self.NF = X.shape

        if max_evidence:
            if self.NS > 4*self.NF:
                self.a, self.b = self.estimate_precision(a, b)
            else:
                self.a, self.b = self.precision(a, b)


    def S_N(self, a, b):
        return np.linalg.inv(a*np.eye(self.NF) + b * np.dot(self.X.T, self.X))

    def m_N(self, a, b):
        return b * np.dot(np.dot(self.S_N(a, b), self.X.T), self.Y)

    def predict(self, x):

        estimate = np.dot(self.m_N(self.a, self.b).T, x.T)
        variance = 1/self.b + np.dot(np.dot(x, self.S_N(self.a, self.b)), x.T)

        return estimate, variance

    def estimate_precision(self, a, b):
        print('NS >> NF: Using maximum evidence approximation.')
        def E_W(w):
            # where w is a column vector
            w = np.vstack(w)
            return .5 * np.dot(w.T, w)[0]

        def E_D(w):
            # where w is a column vector
            w = np.vstack(w)
            ed = 0
            for i in range(self.NS):
                ed += (self.Y[i] - np.dot(w.T, self.X[i, :]))**2
            return .5 * ed[0]

        def system(precision):
            # define precision parameters
            a, b = precision

            # define system of equations

            # eqn for alpha
            eqn_a = a*2*E_W(self.m_N(a, b)) - self.NF

            # eqn for beta
            eqn_b = b*2*E_D(self.m_N(a, b)) - self.NS

            return (eqn_a, eqn_b)

        initial_guess = (a, b)
        a_max, b_max = fsolve(system, initial_guess)

        return a_max, b_max

    def precision(self, a, b):
        print("attempting to maximize evidence function")
        # get eigenvalues of (beta * X.T * X) u = eig * u
        def get_eigenvalues(b):
            eigenvalues, eigenbasis = np.linalg.eig(b*np.dot(self.X.T, self.X))
            return eigenvalues

        def get_gamma(a, b):
            eigenvalues = get_eigenvalues(b)
            gamma = 0
            for eigenvalue in eigenvalues:
                gamma += eigenvalue/(a + eigenvalue)
            return gamma

        def get_lsq(a, b):
            lsq = 0
            for i in range(self.NS):
                lsq += (self.Y[i] - np.dot(self.m_N(a, b), self.X[i, :]))**2
            return lsq

        def system(precision):
            # unpack precision parameters
            a, b = precision

            # define non-linear equations for precision parameters
            eqn_a = get_gamma(a, b) - a * np.dot(self.m_N(a, b), self.m_N(a, b))
            eqn_b = 1/b - (1/(self.NS - get_gamma(a, b))) * get_lsq(a, b)

            return (eqn_a, eqn_b)

        initial_guess = (a, b)
        a_max, b_max = fsolve(system, initial_guess)

        return a_max, b_max

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
