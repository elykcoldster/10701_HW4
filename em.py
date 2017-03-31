from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm

def load_data(filename='X.txt'):
    X = np.loadtxt(filename)
    return (X)

data = load_data('X_new.txt')

gmm = GaussianMixture(n_components=3, covariance_type='diag', tol=1e-3, init_params='random', random_state=123)

gmm.fit(data)

predictions = gmm.predict(data)

pi = (len(np.where(predictions == 0)[0])/3000, len(np.where(predictions == 1)[0])/3000, len(np.where(predictions == 2)[0])/3000)

means = np.zeros((3,5))
covariances = gmm.covariances_

