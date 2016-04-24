import numpy as np
from scipy.stats import multivariate_normal, rv_discrete

def positive_def(a):
    eps = 0.01
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
        try:
            return multivariate_normal.pdf(x, self.mu, self.cov)
        except (ValueError, np.linalg.LinAlgError) as err:
            f = open("error", 'w')
            f.write(str(err) + '\n')
            f.write(str(self.cov) + "\n")
            if 'singular matrix' or 'the input matrix must be positive semidefinite' in err:
                self.cov = positive_def(self.cov)
                f.write(str(self.cov) + "\n\n\n")
                return multivariate_normal.pdf(x, self.mu, self.cov)
            else:
                raise err
