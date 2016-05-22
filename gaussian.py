import numpy as np
from scipy.stats import multivariate_normal

def positive_def(a, i):
    eps = 0.01 * i
    w, v = np.linalg.eigh(a)
    jordan = np.dot(np.transpose(v), a.dot(v))
    di = np.diag_indices(jordan.shape[0])
    jordan = np.diag(jordan[di].clip(eps))
    return v.dot(jordan).dot(np.transpose(v))

class Gauss:
    def __init__(self, m, c):
        if type(m) == np.ndarray:
            self.dim = m.size
        else:
            self.dim = len(m)
        self.mu = m
        self.cov = c

    def density(self, x):
        for i in range(5):
            try:
                return multivariate_normal.pdf(x, self.mu, self.cov)
            except (ValueError, np.linalg.linalg.LinAlgError) as err:
                if 'singular matrix' or 'the input matrix must be positive semidefinite' in err:
                    self.cov = positive_def(self.cov, i)
                    i += 1
                    print("det", np.linalg.det(self.cov))
        return multivariate_normal.pdf(x, self.mu, self.cov)
