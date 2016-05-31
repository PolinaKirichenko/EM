import numpy as np
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from gaussian import Gauss
from true import real_params
import math
import bisect
import sys
from sklearn.utils.extmath import logsumexp
import scipy


def positive_def(a):
    eps = 0.001
    w, v = np.linalg.eigh(a)
    jordan = np.dot(np.transpose(v), a.dot(v))
    di = np.diag_indices(jordan.shape[0])
    jordan = np.diag(jordan[di].clip(eps))
    return v.dot(jordan).dot(np.transpose(v))


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1):
    """Log probability for full covariance matrices."""
    # n_samples = 1 #
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
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
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


class GMM:
    def Estep(self, train_set):
        lpr = (_log_multivariate_normal_density_full(train_set, self.means, self.covs,) + np.log(self.w))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities.T

    def Mstep(self, minibatch, gamma_diff, exnum_old, temp, tss):
        bs = minibatch.shape[0]
        exnum = exnum_old + gamma_diff.sum(axis=1)
        self.w = exnum / tss
        for i in range(self.compnum):
            mu_old = np.copy(self.means[i])
            self.means[i] += np.dot(gamma_diff[i], minibatch - self.means[i]) / exnum[i]
            
            centre = minibatch - self.means[i]
            temp.fill(0)
            for n in range(bs):
               temp += gamma_diff[i][n] * (np.asmatrix(centre[n]).T * centre[n] - self.covs[i])
            temp += exnum_old[i] * (np.asmatrix(self.means[i] - mu_old)).T * (self.means[i] - mu_old)
            temp /= exnum[i]
            self.covs[i] += temp
        return exnum


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

    def stochEM(self, train_set):
        s = 1
        tss = train_set.shape[0]
        gamma_new = np.zeros((self.compnum, s)) # for mini-batch
        complete_gamma = np.full((self.compnum, tss), 1 / self.compnum)
        exnum = complete_gamma.sum(axis=1)
        
        temp = np.zeros((self.dim, self.dim))

        for i in range(tss):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]
            logls, gamma_new = self.Estep(minibatch)
            exnum = self.Mstep(minibatch, gamma_new - complete_gamma[:, batch_idx], exnum, temp, tss)
            complete_gamma[:, batch_idx] = gamma_new

            ll, _ = self.Estep(train_set)
            print(ll.sum())

            if i % 50 == 0:
                draw(train_set, self, i)



def closest(obs, point):
    idx = (np.linalg.norm(obs - point, axis=1)).argmin()
    return idx, obs[idx]

def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.full((gnum, ), 1 / gnum)

    #mu = np.random.random_sample((gnum, dim)) * (np.amax(obs, axis=0) - np.amin(obs, axis=0)) + np.amin(obs, axis=0)
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


def draw(obs, model, j):
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
    for i in range(len(model.means)):
        rv = multivariate_normal(model.means[i], model.covs[i])
        plt.contour(X, Y, model.w[i] * rv.pdf(pos), zorder=2)
        plt.scatter(model.means[i][0], model.means[i][1], color='r', s=20, zorder=2)
    
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.grid(which='both')
    plt.savefig('increment/inc' + str(j) + '.png')
    plt.close()


def test():
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    setnum = int(sys.argv[1])
    gnum = comp[setnum]
    obs = np.loadtxt("data/" + "input" + str(setnum))

    real_gauss = real_params(setnum)
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    values = []
    mean_err = []
    for i in range(30):
        print(i)
        w, mu, cov = initParam(gnum, obs)
        model.w = w
        for i in range(gnum):
            model.means[i] = mu[i]; model.covs[i] = cov[i]

        logll = model.stochEM(obs)
        values.append(logll)

        for i in range(gnum):
            idx = (np.linalg.norm(model.means - real_gauss.mu[i], axis=1)).argmin()
            mean_err.append(np.linalg.norm(model.means[idx] - real_gauss.mu[i]))

    np.savetxt("report/inc/mean" + str(setnum), np.array(mean_err))
    np.savetxt("report/inc/ll" + str(setnum), np.array(values))


def main():
    # mini-batch size = 1
    setnum = int(sys.argv[1])
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))
    pics = (obs.shape[1] == 2)
    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.stochEM(obs)
    model.printParam()

    real_gauss = real_params(setnum)
    means = []
    covs = []

    for i in range(gnum):
        idx = (np.linalg.norm(model.means - real_gauss.mu[i], axis=1)).argmin()
        means.append(np.linalg.norm(model.means[idx] - real_gauss.mu[i]))
        covs.append(np.linalg.norm(np.diagonal(model.covs[idx]) - np.diagonal(real_gauss.cov[i])))

    print(means)
    print(covs)

    if pics:
        draw(obs, model, "fin")

main()
