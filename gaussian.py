import numpy as np
from scipy.stats import multivariate_normal, rv_discrete

class Gauss:
    def __init__(self, m, c):
        self.dim = m.size
        self.mu = m
        self.cov = c

    def density(self, x):
        #try:
        return multivariate_normal.pdf(x, self.mu, self.cov)
        #except ValueError:
        #    self.cov = np.eye(self.dim)
