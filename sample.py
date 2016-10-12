import numpy as np
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from true_params import real_params

def generateSamples(w, mu, cov, s):
    dim = len(mu[0])
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
