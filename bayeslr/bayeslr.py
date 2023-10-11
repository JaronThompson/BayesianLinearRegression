import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial


# define JIT compiled functions

@jit
def GMG(G, M):
    # return G @ M @ G.T
    return jnp.einsum("ni,ij,nj", G, M, G)


@jit
def update_covariance(alpha, beta, X):
    return jnp.linalg.inv(alpha * jnp.eye(len(alpha)) + beta * jnp.einsum("ni,nj->ij", X, X))


@jit
def update_params(Ainv, beta, X, Y):
    # return jnp.einsum("ij,j->i", Ainv, beta * jnp.nansum(jnp.einsum("nj,n->nj", X, Y), 0))
    return beta*jnp.einsum('ij,nj,n->i', Ainv, X, jnp.nan_to_num(Y))


# jit compile function to compute log of determinant of a matrix
@jit
def log_det(A):
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))


# jit compile prediction covariance computation
@jit
def compute_predCOV(BetaInv, G, Ainv):
    return BetaInv + jnp.einsum("ni,ij,nj->n", G, Ainv, G)


# linear regression class

class LR():

    def __init__(self, alpha=1., beta=1.):

        # initialize hyper-parameters
        self.alpha = alpha
        self.beta = beta
        self.a = 1e-4
        self.b = 1e-4

    # function to predict mean of outcomes
    @partial(jit, static_argnums=(0,))
    def forward(self, X, params):
        # make point predictions
        return X @ params

    # estimate posterior parameter distribution
    def fit(self, X, Y, evd_tol=1e-3, patience=1):

        # number of basis functions
        self.n_params = X.shape[-1]

        # reshape Y
        Y = Y.ravel()

        # init convergence metrics
        self.itr = 0
        passes = 0
        fails = 0
        previdence = -np.inf

        # init convergence status
        converged = False

        # initialize hyper parameters
        self.init_hypers(X, Y)

        while not converged:
            # update Alpha and beta hyper-parameters
            if self.itr > 0: self.update_hypers(X, Y)

            # compute hessian inverse
            self.Ainv = update_covariance(self.alpha, self.beta, X)

            # pseudo inverse to compute params (ignoring NaN values)
            self.params = update_params(self.Ainv, self.beta, X, Y)
            self.objective(self.params, X, Y)

            # update evidence
            self.update_evidence()
            print("Evidence {:.3f}".format(self.evidence))

            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1., np.abs(self.evidence)])

            # update pass count
            if convergence < evd_tol:
                passes += 1
                print("Pass count ", passes)
            else:
                passes = 0

            # increment fails if convergence is negative
            if self.evidence < previdence:
                fails += 1
                print("Fail count ", fails)

            # determine whether algorithm has converged
            if passes >= patience:
                converged = True

            # update evidence
            previdence = np.copy(self.evidence)
            self.itr += 1

    def callback(self, xk, res=None):
        print("Loss: {:.3f}".format(self.loss))
        return True

    # function to compute NLL loss function
    def compute_NLL(self, params, X, Y, beta):
        outputs = self.forward(X, params)
        error = jnp.nan_to_num(outputs - Y)
        return beta * jnp.sum(error ** 2) / 2.

    # define objective function
    def objective(self, params, X, Y):
        # init loss with parameter penalty
        self.loss = jnp.dot(self.alpha * params, params) / 2.

        # forward pass
        self.loss += self.compute_NLL(params, X, Y, self.beta)

        return self.loss

    # update hyperparameters alpha and beta
    def init_hypers(self, X, Y):
        # compute number of independent samples in the data
        self.N = np.sum(~np.isnan(Y), 0)

        # init alpha
        self.alpha = self.alpha * jnp.ones(self.n_params)

        # update beta
        self.beta = 1.

    # update hyperparameters alpha and beta
    def update_hypers(self, X, Y):

        # forward
        outputs = self.forward(X, self.params)
        error = jnp.nan_to_num(outputs - Y)

        # sum of measurement covariance update
        yCOV = np.sum(error ** 2) + GMG(X, self.Ainv)

        # update alpha option 1: unique alpha for each parameter
        self.alpha = 1. / (self.params ** 2 + jnp.diag(self.Ainv) + 2. * self.a)

        # update alpha option 2: isotropic Gaussian prior
        # alpha = self.n_params / (jnp.sum(self.params**2) + jnp.trace(self.Ainv) + 2.*self.a)
        # self.alpha = alpha*jnp.ones_like(self.params)

        # divide by number of observations
        yCOV = yCOV / self.N

        # update beta
        self.beta = 1. / (yCOV + self.b)

    # compute the log marginal likelihood
    def update_evidence(self):
        # compute evidence
        self.evidence = 1 / 2 * np.sum(self.N * np.log(self.beta)) + \
                        1 / 2 * np.nansum(np.log(self.alpha)) + \
                        1 / 2 * log_det(self.Ainv) - self.loss

    # function to predict mean and stdv of outcomes
    def predict(self, X):

        # point estimates
        preds = self.forward(X, self.params)

        # compute covariances
        COV = compute_predCOV(1. / self.beta, X, self.Ainv)

        # pull out standard deviations
        stdvs = np.sqrt(COV)

        return preds, stdvs
