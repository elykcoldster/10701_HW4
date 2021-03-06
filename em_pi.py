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

pi = np.random.rand(3)
pi = pi/sum(pi)

print('Initial class probabilities:')
print(pi)

covariances = gmm.covariances_

thresh = 0.001
delta = np.inf
iters = 0

means = []
pis = []

np.random.seed(123)
random_init = np.random.randint(0, high=data.shape[0], size=3)
means.append(data[random_init])
pis.append(np.copy(pi))

while delta > thresh:
	r = np.zeros((data.shape[0], 3))
	
	for i in range(0, data.shape[0]):	
		denom = 0
		for j in range(0,3):
			denom += pi[j]*g(data[i], means[iters][j,:], covariances[j])
		for k in range(0,3):
			r[i,k]=pi[k]*g(data[i], means[iters][k,:], covariances[k])/denom

	psums = np.zeros(3)
	for k in range(0,3):
		for i in range(0,data.shape[0]):
			psums[k] += r[i,k]

	mus = np.zeros((3,5))
	for k in range(0,3):
		for i in range(0,data.shape[0]):
			mus[k,:] += data[i] * r[i,k]/ psums[k]

	for k in range(0,3):
		pi[k] = psums[k]/data.shape[0]

	means.append(mus)
	pis.append(np.copy(pi))

	iters += 1
	if iters > 0:
		means_now = means[iters]
		means_prev = means[iters - 1]

		pis_now = pis[iters]
		pis_prev = pis[iters - 1]

		del_means = abs(means_now - means_prev)
		del_pis = abs(pis_now - pis_prev)

		delta = max(np.max(del_pis), np.max(del_means))
print('Means')
print(means[iters])
print('Class probabilities')
print(pi, sum(pi))
print('GaussianMixture means')
print(gmm.means_)