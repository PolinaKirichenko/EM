import numpy as np
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def generateSamples(w, mu, cov, s):
    dim = mu.shape[1]
    d = rv_discrete(values = (range(len(w)), w))
    components = d.rvs(size=s)
    # generate samples of size of each component, then shuffle
    if dim > 1:
        return components, np.array([np.random.multivariate_normal(mu[i], cov[i], 1)[0] for i in components])
    else:
        return components, np.asmatrix([np.random.normal(mu[i], cov[i], 1)[0] for i in components]).T

def generate(w, mu, cov, n, idx):
    comps, obs = generateSamples(w, mu, cov, n)
    np.savetxt("data/input" + str(idx), obs, delimiter='\t', fmt='%f')
    return obs

def main():
    idx = 101
    gnum = 3
    w = [0.33, 0.32, 0.35]

    mu = np.zeros((gnum, 100))
    mu[1] = np.full((1, 100), 5, dtype=float)
    mu[2] = np.full((1, 100), 10, dtype=float)

    cov = np.full((gnum, 100, 100), np.eye(100))

    obs = generate(w, mu, cov, 5000, idx)
    #plt.plot(*zip(*obs), marker='o', ls='', zorder=1)
    #plt.savefig('sets/set' + str(idx) + '.png')

main()