'''
LR.py by Jaron Thompson

This script includes the BLR class for implementing Bayesian linear regression.
Precision hyper-parameters are updated based on the training data by maximizing
the evidence function using the Expecation-Maximization algorithm.

Also included is a GLR class for general linear regression.

'''
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, jit

### Bayesian Linear Regression model ###

class BLR:

    def __init__(self, X, Y, alpha=1e-3, beta=1.0, tol=1e-3):
        self.X = X
        self.Y = Y.ravel()

        # number of samples and basis functions
        self.n_samples, self.n_basis = X.shape

        # initialize model parameters
        self.mu = np.zeros(self.n_basis)

        # hyper parameters
        self.alpha = alpha*np.ones(self.n_basis)
        self.A = np.diag(self.alpha)
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
            self.A = self.alpha*np.eye(self.n_basis) + self.beta*self.X.T@self.X
            self.Ainv = np.linalg.inv(self.A)
            self.Ainv = 1/2*(self.Ainv + self.Ainv.T)
            self.mu = self.beta*self.Ainv@self.X.T@self.Y
            Y_pred = self.X@self.mu
            self.SSE = np.sum(np.nan_to_num(Y_pred - self.Y)**2)

            # M step
            gamma = np.sum(1. - self.alpha*np.diag(self.Ainv))
            # gamma = self.n_basis - self.alpha*np.trace(self.Ainv)
            self.alpha = np.ones(self.n_basis) / (self.mu**2 + np.diag(self.Ainv) + self.a)
            # self.alpha = (self.n_basis) / (np.dot(self.mu, self.mu) + np.trace(self.Ainv) + self.a)
            self.beta  = (self.n_samples) / (self.SSE + gamma/self.beta + self.a)

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

    def fit_MCMC(self, num_warmup=500, num_samples=1000, rng_key=0):
        # define probabilistic model
        predict = jit(lambda X,mu: X@mu)

        def model(X, y):
            # parameter random variable
            mu = numpyro.sample('mu', dist.MultivariateNormal(loc=self.mu, covariance_matrix=self.Ainv))
            # likelihood
            L = numpyro.sample('y', dist.Normal(loc=predict(X,mu), scale=(1./self.beta)**.5), obs=y)
        # instantiate MCMC object with NUTS kernel
        self.mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples)
        self.mcmc.warmup(random.PRNGKey(rng_key), self.X, self.Y) #, init_params=self.mu)
        self.mcmc.run(random.PRNGKey(rng_key), self.X, self.Y) #, init_params=self.mu)
        # save posterior samples
        self.posterior_params = np.array(self.mcmc.get_samples()['mu'])

    def predict(self, X):
        y_pred = X@self.mu
        y_var  = 1/self.beta + np.einsum('ni,ij,nj->n', X, self.Ainv, X)
        return y_pred, np.sqrt(y_var)

    def predict_MCMC(self, X):
        y_preds  = np.einsum('ni,si->sn', X, self.posterior_params)
        y_preds += np.random.randn(y_preds.shape[0], y_preds.shape[1])*(1/self.beta)**.5
        return np.mean(y_preds, 0), np.std(y_preds, 0)

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
