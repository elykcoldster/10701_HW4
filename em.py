from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import random
def load_data(filename='X.txt'):
    X = np.loadtxt(filename)
    return (X)

def g(x, u, var):
	return 1/(np.sqrt(2*np.pi*var))*np.exp(-np.linalg.norm(x-u)**2/(2*var))

data = load_data('X_new.txt')

gmm = GaussianMixture(n_components=3, covariance_type='spherical', tol=1e-3, init_params='random', random_state=123)

gmm.fit(data)

predictions = gmm.predict(data)

pi = (len(np.where(predictions == 0)[0])/3000, len(np.where(predictions == 1)[0])/3000, len(np.where(predictions == 2)[0])/3000)

covariances = gmm.covariances_

thresh = 0.001
delta = np.inf
iters = 0

means = []

random_init = np.random.randint(0, high=data.shape[0], size=3)
means.append(data[random_init])

while delta > thresh:
	psums = np.zeros(3)
	for i in range(0,3):
		for x in data:
			psums[i] += pi[i]*g(x, means[iters][i,:], covariances[i])

	musums = np.zeros((3,5))
	for i in range(0,3):
		for x in data:
			musums[i,:] += x * pi[i]*g(x, means[iters][i,:], covariances[i]) / psums[i]

	means.append(musums)
	iters += 1
	if iters > 0:
		means_now = means[iters]
		means_prev = means[iters - 1]
		del_means = abs(means_now - means_prev)
		delta = np.max(del_means)
print(means[iters])
print(gmm.means_)