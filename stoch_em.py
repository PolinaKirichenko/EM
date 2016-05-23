import sys
import numpy as np
import bisect
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from gaussian import Gauss
from true import real_params
import math
from sklearn.utils.extmath import logsumexp
import scipy


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-300):
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
                cv_chol = scipy.linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except scipy.linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = scipy.linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        chmatr.append(cv_chol)
        chsol.append(cv_sol)
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob, chmatr, chsol


class GMM:
    def Estep(self, train_set):
        lpr, chmatr, chsol = _log_multivariate_normal_density_full(train_set, self.means, self.covs,) 
        lpr += np.log(self.w)
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities.T, chmatr, chsol

    def Mstep(self, minibatch, gamma, chmatr, chsol, lr):
        # bs = minibatch.shape[0] = 1
        #self.w += lr * gamma.sum(axis=1) / self.w
        #self.w /= self.w.sum()
        for i in range(self.compnum):
            inv_chol = scipy.linalg.solve_triangular(chmatr[i], np.eye(self.dim), lower=True)
            self.means[i] += lr * np.dot(np.dot(inv_chol.T, inv_chol), np.dot(gamma[i], minibatch - self.means[i]))            
            self.covs[i] += lr * gamma[i][0] / 2 * np.dot(inv_chol.T, np.dot(-np.eye(self.dim) + chsol[i].T * chsol[i], inv_chol))

        
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

    def stochEM(self, train_set, const):
        # Assume mini-batch size is 1 #
        s = 1
        tss = train_set.shape[0]
        pics = (train_set.shape[1] == 2)
        gamma = np.zeros((self.compnum, s)) # for mini-batch

        for i in range(50):
            batch_idx = np.random.randint(train_set.shape[0], size=1)
            minibatch = train_set[batch_idx, :]
            logls, gamma, chmatr, chsol = self.Estep(minibatch)
            self.Mstep(minibatch, gamma, chmatr, chsol, const)

        for i in range(50, 2 * tss):
            batch_idx = np.random.randint(train_set.shape[0], size=1)
            minibatch = train_set[batch_idx, :]
            logls, gamma, chmatr, chsol = self.Estep(minibatch)
            self.Mstep(minibatch, gamma, chmatr, chsol, const / math.sqrt(i + 1))


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
    plt.savefig('stochastic/stoch' + str(j) + '.png')
    plt.close()


def find_lr():
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    setnum = int(sys.argv[1])
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))
    real_gauss = real_params(setnum)

    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)

    f1 = open("report/const_sqrt/mean13_0.05", 'ab+')
    f2 = open("report/const_sqrt/ll13_0.05", 'a+')

    poss = [0.05]
    for const in poss:
        values = []
        mean_err = []
        for i in range(9):
            print(i)
            w, mu, cov = initParam(gnum, obs)
            model.w = w
            for i, g in enumerate(model.gaussian):
                g.mu = mu[i]; g.cov = cov[i]
            try:
                logll = model.stochEM(obs, const)
                values.append(logll)
            except Exception as e:
                print(e)
                f1.close()
                f2.close()
                exit(0)

            all_mu = np.array([g.mu for g in model.gaussian])
            for i in range(gnum):
                idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
                mean_err.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))

            np.savetxt(f1, np.array(mean_err[len(mean_err) - gnum:]))
            f2.write(str(logll) + '\n')
        # np.savetxt("report/const_sqrt/mean" + str(setnum) + "_" + str(const), np.array(mean_err))
        # np.savetxt("report/const_sqrt/ll" + str(setnum) + "_" + str(const), np.array(values))
        print(const)
    f1.close()
    f2.close()



def test_accuracy(const, fnum, out, gnum):
    obs = np.loadtxt("data/" + "input" + str(fnum))
    real_gauss = real_params(fnum)

    means = []
    covs = []
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    for j in range(10):
        print(j)
        model.w = w
        for i, g in enumerate(model.gaussian):
            g.mu = mu[i]; g.cov = cov[i]
        model.stochEM(obs, const)
        all_mu = np.array([g.mu for g in model.gaussian])
        all_cov = np.array([g.cov for g in model.gaussian])
        for i in range(gnum):
            idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
            means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
            covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    out.write("Testing set " + str(fnum) + '\n')
    out.write("mu " + str(sum(means) / len(means)) + '\n' + "cov " + str(sum(covs) / len(covs)) + '\n\n')


def report():
    setnum = int(sys.argv[1])
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))
    out = open("report/stoch/report.txt", 'a+')

    lr_const = 0.03

    real_gauss = real_params(setnum)
    means = []
    covs = []
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)

    for j in range(20):
        print(j)
        w, mu, cov = initParam(gnum, obs)
        model.w = w
        for i, g in enumerate(model.gaussian):
            g.mu = mu[i]; g.cov = cov[i]
        model.stochEM(obs, lr_const)
        all_mu = np.array([g.mu for g in model.gaussian])
        all_cov = np.array([g.cov for g in model.gaussian])
        for i in range(gnum):
            idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
            means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
            covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    np.savetxt("report/stoch/res" + str(setnum), (np.array(means), np.array(covs)))
    out.write("Testing set " + str(setnum) + '\n')
    out.write("mu " + str(sum(means) / len(means)) + '\n' + "cov " + str(sum(covs) / len(covs)) + '\n\n')


def em_train():
    setnum = int(sys.argv[1])
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))
    
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    model.stochEM(obs, 0.05)
    draw(obs, model, "fin")

    real_gauss = real_params(setnum)
    means = []
    covs = []

    for i in range(gnum):
        idx = (np.linalg.norm(model.means - real_gauss.mu[i], axis=1)).argmin()
        means.append(np.linalg.norm(model.means[idx] - real_gauss.mu[i]))
        covs.append(np.linalg.norm(np.diagonal(model.covs[idx]) - np.diagonal(real_gauss.cov[i])))

    print(means)
    print(covs)

em_train()
