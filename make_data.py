import numpy as np
import sklearn.datasets

X, y = sklearn.datasets.make_regression(n_samples=200, n_features=150,
    n_informative=75, n_targets=1, noise=25)

np.savetxt('features.csv', X)
np.savetxt('targets.csv', y)
