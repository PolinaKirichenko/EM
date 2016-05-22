import numpy as np
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time
import bisect
from true import real_params
from gaussian import Gauss
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.utils.extmath import logsumexp





class GMM:
    def Estep(self, train_set, norm, gamma):
        tss = train_set.shape[0]
        prob = np.dot(self.w, norm) # prob[j] probability of observation j

        np.copyto(gamma, norm)
        gamma *= self.w[:, np.newaxis]
        gamma /= prob

    def Mstep(self, train_set, gamma):
        tss = train_set.shape[0]
        exnum = gamma.sum(axis=1)
        self.w = exnum / tss
        for i in range(self.compnum):
            self.gaussian[i].mu = np.dot(gamma[i], train_set) / exnum[i]
            centre = train_set - self.gaussian[i].mu

            self.gaussian[i].cov.fill(0)
            for n in range(tss):
                self.gaussian[i].cov += gamma[i][n] * np.asmatrix(centre[n]).T * centre[n]
            self.gaussian[i].cov /= exnum[i]

    def updateProbabilities(self, norm, train_set):
        for i in range(self.compnum):
            norm[i] = self.gaussian[i].density(train_set) + 1e-300

    def logLikelihood(self, train_set, norm):
        tss = train_set.shape[0]
        ll = 0
        prob = np.dot(self.w, norm)
        for i in range(tss):
            ll += log(prob[i])
        return ll

        
    def __init__(self, w, mu, cov):
        self.dim = mu[0].size # dimension
        self.compnum = w.size # the number of gaussian
        self.w = w # weights of gaussian
        self.gaussian = [Gauss(mu[i], cov[i]) for i in range(self.compnum)]

    def printParam(self):
        print(self.w)
        for g in self.gaussian:
            print(g.mu)
            print(g.cov, '\n')

    def EM(self, train_set):
        tss = train_set.shape[0]
        gamma = np.zeros((self.compnum, tss))
        norm = np.zeros((self.compnum, tss))
        self.updateProbabilities(norm, train_set)
        print(self.logLikelihood(train_set, norm))

        for i in range(25):
            self.Estep(train_set, norm, gamma)
            self.Mstep(train_set, gamma)
            self.updateProbabilities(norm, train_set)
            print(self.logLikelihood(train_set, norm))


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
    for i, g in enumerate(model.gaussian):
        rv = multivariate_normal(g.mu, g.cov)
        plt.contour(X, Y, model.w[i] * rv.pdf(pos), zorder=2)
        plt.scatter(g.mu[0], g.mu[1], color='r', s=20, zorder=2)
    
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.grid(which='both')
    plt.savefig('em' + str(j) + '.png')
    plt.close()


def main():
    setnum = 12

    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))

    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.EM(obs)
    #model.printParam()
    #draw(obs, model, "fin")

    real_gauss = real_params(setnum)
    means = []
    covs = []

    all_mu = np.array([g.mu for g in model.gaussian])
    all_cov = np.array([g.cov for g in model.gaussian])
    for i in range(gnum):
        idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
        means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
        covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    print(means)
    print(covs)

main()
