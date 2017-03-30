from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename='X.txt'):
    X = np.loadtxt(filename)
    return (X)

data = load_data('X_new.txt')

gmm = GaussianMixture(n_components=3, covariance_type='diag', tol=1e-3, init_params='random', random_state=123)

gmm.fit(data)

print("Means:")
print(gmm.means_)
print("Covariances")
print(gmm.covariances_)

clusters = gmm.predict(data)
xi = (data[np.where(clusters == 0)], data[np.where(clusters == 1)], data[np.where(clusters == 2)])

plt.title('First two dimensions ($x_{i1}$ and $x_{i2}$) with clustering')
plt.xlabel(r'$x_{i1}$')
plt.ylabel(r'$x_{i2}$')
plt.plot(xi[0][:,0], xi[0][:,1], '.', color='blue')
plt.plot(xi[1][:,0], xi[1][:,1], '.', color='green')
plt.plot(xi[2][:,0], xi[2][:,1], '.', color='red')
plt.show()

plt.title('$x_{i3}$ and $x_{i4}$ with clustering')
plt.xlabel(r'$x_{i3}$')
plt.ylabel(r'$x_{i4}$')
plt.plot(xi[0][:,2], xi[0][:,3], '.', color='blue')
plt.plot(xi[1][:,2], xi[1][:,3], '.', color='green')
plt.plot(xi[2][:,2], xi[2][:,3], '.', color='red')
plt.show()

plt.title('$x_{i4}$ and $x_{i5}$ with clustering')
plt.xlabel(r'$x_{i4}$')
plt.ylabel(r'$x_{i5}$')
plt.plot(xi[0][:,3], xi[0][:,4], '.', color='blue')
plt.plot(xi[1][:,3], xi[1][:,4], '.', color='green')
plt.plot(xi[2][:,3], xi[2][:,4], '.', color='red')
plt.show()