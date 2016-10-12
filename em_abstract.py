import sys
import numpy as np
import bisect
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from gaussian import Gauss
from sklearn.utils.extmath import logsumexp
import scipy

def positive_def(a):
    eps = 0.01
    w, v = np.linalg.eigh(a)
    jordan = np.dot(np.transpose(v), a.dot(v))
    di = np.diag_indices(jordan.shape[0])
    jordan = np.diag(jordan[di].clip(eps))
    return v.dot(jordan).dot(np.transpose(v))


### This function is taken from scikit-learn GMM. ###
def _log_multivariate_normal_density_full(X, means, covars, min_covar=1):
    """Log probability for full covariance matrices."""
    # n_samples = 1 #
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    chmatr = []
    chsol = []
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = scipy.linalg.cholesky(cv, lower=True)
        except scipy.linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                #cv_chol = scipy.linalg.cholesky(cv + min_covar * np.eye(n_dim),
                #                          lower=True)
                cv_chol = scipy.linalg.cholesky(positive_def(cv), lower=True)
            except scipy.linalg.LinAlgError:
                np.savetxt("errorch", cv + min_covar * np.eye(n_dim))
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = scipy.linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        chmatr.append(cv_chol)
        chsol.append(cv_sol)
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob, chmatr, chsol

def closest(obs, point):
    idx = (np.linalg.norm(obs - point, axis=1)).argmin()
    return idx, obs[idx]

def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.full((gnum, ), 1 / gnum)
    
    mu = np.random.multivariate_normal(obs.mean(axis=0), np.diag(obs.std(axis=0) ** 2), gnum)
    full = list(range(obs.shape[0]))
    idxs = []
    for i in range(gnum):
        idx, mu[i] = closest(obs[list(set(full) - set(idxs)), :], mu[i])
        for i in idxs:
            if i <= idx:
                idx += 1
            else:
                break
        bisect.insort(idxs, idx)

    cov = np.full((gnum, dim, dim), np.eye(dim, dtype = float))
    return w, mu, cov

class AbstractGMM:
    def __init__(self, w, mu, cov):
        self.dim = mu[0].size # dimension
        self.compnum = w.size # the number of gaussian
        self.w = w # weights of gaussian
        self.means = mu
        self.covs = cov

    def printParam(self):
        print(self.w)
        for i in range(self.compnum):
            print(self.means[i])
            print(self.covs[i], '\n')

    def draw_2dim(self, obs, path):
        if self.dim != 2:
            print("Error: plots only for 2-dimensional data.")
            return
        minorLocator = MultipleLocator(1)
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(*zip(*obs), marker='o', ls='', zorder=1)
        delta = 0.1
        obsmax = np.amax(obs, axis=0)
        obsmin = np.amin(obs, axis=0)
        X, Y = np.mgrid[obsmin[0]:obsmax[0]:delta, obsmin[1]:obsmax[1]:delta] 
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        for i in range(len(self.means)):
            rv = multivariate_normal(self.means[i], self.covs[i])
            plt.contour(X, Y, self.w[i] * rv.pdf(pos), zorder=2)
            plt.scatter(self.means[i][0], self.means[i][1], color='r', s=20, zorder=2)
        
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        plt.grid(which='both')
        plt.savefig(path)
        plt.close()

    def Estep(self, train_set):
        lpr, chmatr, chsol = _log_multivariate_normal_density_full(train_set, self.means, self.covs,) 
        lpr += np.log(self.w)
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities.T, chmatr, chsol

    def EM(self, train_set):
        pass
