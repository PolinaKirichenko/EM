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

class Gauss:
    def __init__(self, m, c):
        self.dim = m.size
        self.mu = m
        self.cov = c

    def density(self, x):
        return multivariate_normal.pdf(x, self.mu, self.cov)


class GMM:
    def Estep(self, train_set, norm, gamma_new):
        prob = np.dot(self.w, norm) # prob[j] probability of observation j
        np.copyto(gamma_new, norm)
        gamma_new *= self.w[:, np.newaxis]
        gamma_new /= prob

    def Mstep(self, minibatch, gamma_diff, exnum_old, temp, tss):
        bs = minibatch.shape[0]
        exnum = exnum_old + gamma_diff.sum(axis=1)
        self.w = exnum / tss
        for i in range(self.compnum):
            mu_old = self.gaussian[i].mu
            self.gaussian[i].mu += np.dot(gamma_diff[i], minibatch - self.gaussian[i].mu) / exnum[i]
            centre = minibatch - self.gaussian[i].mu

            temp.fill(0)
            for n in range(bs):
                temp += gamma_diff[i][n] * (np.asmatrix(centre[n]).T * centre[n] - self.gaussian[i].cov)
            temp += exnum_old[i] * (np.asmatrix(self.gaussian[i].mu - mu_old)).T * (self.gaussian[i].mu - mu_old)
            temp /= exnum_old[i]
            self.gaussian[i].cov += temp


    def updateProbabilities(self, norm, train_set):
        for i in range(self.compnum):
            norm[i] = self.gaussian[i].density(train_set) + 1e-320

    def logLikelihood(self, train_set, norm):
        tss = train_set.shape[0]
        ll = 0
        prob = np.dot(self.w, norm)
        for i in range(tss):
            ll += log(prob[i])
        return ll

    def initialExnum(self, complete_norm):
        gamma = np.empty([complete_norm.shape[0], complete_norm.shape[1]])
        prob = np.dot(self.w, complete_norm)
        np.copyto(gamma, complete_norm)
        gamma *= self.w[:, np.newaxis]
        gamma /= prob
        return gamma.sum(axis=1)

        
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

    def stochEM(self, train_set, s):
        tss = train_set.shape[0]
        gamma_new = np.zeros((self.compnum, s)) # for mini-batch
        complete_gamma = np.zeros((self.compnum, tss))

        norm = np.zeros((self.compnum, s)) # for mini-batch
        # complete_norm = np.zeros((self.compnum, tss))

        self.updateProbabilities(complete_norm, train_set)
        exnum = self.initialExnum(complete_norm)
        temp = np.zeros((self.dim, self.dim))

        # ll = self.logLikelihood(train_set, complete_norm)
        for i in range(tss):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]
            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma_new)
            self.Mstep(minibatch, gamma_new - complete_gamma[:, batch_idx], exnum, temp, tss)

            # self.updateProbabilities(complete_norm, train_set)
            # ll = self.logLikelihood(train_set, complete_norm)

            #print(ll)
            #if i % 200 == 0:
            #    draw(train_set, self, i)


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
# Here I assume that dimension is 2 #
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
    plt.savefig('increment/stoch' + str(j) + '.png')
    plt.close()


def test():
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    setnum = int(sys.argv[1])
    gnum = comp[setnum]
    obs = np.loadtxt("data/" + "input" + str(setnum))
    out = open("incr_tot", 'a+')

    real_gauss = real_params(setnum)
    means = []
    covs = []
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    for j in range(15):
        print(j)
        w, mu, cov = initParam(gnum, obs)
        model.w = w
        for i, g in enumerate(model.gaussian):
            g.mu = mu[i]; g.cov = cov[i]
        model.stochEM(obs, 1)
        all_mu = np.array([g.mu for g in model.gaussian])
        all_cov = np.array([g.cov for g in model.gaussian])
        for i in range(gnum):
            idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
            means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
            covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    np.savetxt("incr" + str(setnum), (np.array(means), np.array(covs)))
    out.write("Testing set " + str(setnum) + '\n')
    out.write("mu " + str(sum(means) / len(means)) + '\n' + "cov " + str(sum(covs) / len(covs)) + '\n\n')
    out.close()


def main():
    # mini-batch size = 1
    setnum = 8
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))
    pics = (obs.shape[1] == 2)
    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.printParam()
    if pics:
        draw(obs, model, 0)
    model.stochEM(obs, 1)
    model.printParam()
    if pics:
        draw(obs, model, "fin")

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

test()
