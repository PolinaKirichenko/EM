import numpy as np
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time

class Gauss:
    def __init__(self, m, c):
        self.dim = m.size
        self.mu = m
        self.cov = c

    def density(self, x):
        return multivariate_normal.pdf(x, self.mu, self.cov)


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
            self.gaussians[i].mu = np.dot(gamma[i], train_set) / exnum[i]
            centre = train_set - self.gaussians[i].mu

            self.gaussians[i].cov.fill(0)
            for n in range(tss):
                self.gaussians[i].cov += gamma[i][n] * np.asmatrix(centre[n]).T * centre[n]
            self.gaussians[i].cov /= exnum[i]

    def updateProbabilities(self, norm, train_set):
        for i in range(self.compnum):
            norm[i] = self.gaussians[i].density(train_set)

    def logLikelihood(self, train_set, norm):
        tss = train_set.shape[0]
        ll = 0
        prob = np.dot(self.w, norm)
        for i in range(tss):
            ll += log(prob[i])
        return ll

        
    def __init__(self, w, mu, cov):
        self.dim = mu[0].size # dimension
        self.compnum = w.size # the number of gaussians
        self.w = w # weights of gaussians
        self.gaussians = [Gauss(mu[i], cov[i]) for i in range(self.compnum)]

    def printParam(self):
        print(self.w)
        for g in self.gaussians:
            print(g.mu)
            print(g.cov, '\n')

    def EM(self, train_set, iter):
        tss = train_set.shape[0]
        gamma = np.zeros((self.compnum, tss))
        norm = np.zeros((self.compnum, tss))
        self.updateProbabilities(norm, train_set)
        print(self.logLikelihood(train_set, norm))

        for i in range(iter):
            self.Estep(train_set, norm, gamma)
            self.Mstep(train_set, gamma)
            self.updateProbabilities(norm, train_set)
            print(self.logLikelihood(train_set, norm))


def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.random.random(gnum)
    w /= w.sum()

    mu = np.random.multivariate_normal(obs.mean(axis=0), np.diag(obs.std(axis=0)), gnum)
    #mu = np.random.random_sample((gnum, dim)) * (np.amax(obs, axis=0) - np.amin(obs, axis=0)) + np.amin(obs, axis=0)
    
    cov = np.full((gnum, dim, dim), np.eye(dim, dtype = float))
    return w, mu, cov

def generateSamples(w, mu, cov, s):
    dim = len(mu[0])
    d = rv_discrete(values = (range(len(w)), w))
    components = d.rvs(size=s)
    # generate samples of size of each component, then shuffle
    if dim > 1:
        return components, np.array([np.random.multivariate_normal(mu[i], cov[i], 1)[0] for i in components])
    else:
        return components, np.asmatrix([np.random.normal(mu[i], cov[i], 1)[0] for i in components]).T

def generate(true_w, true_mu, true_cov, n):
    comps, obs = generateSamples(true_w, true_mu, true_cov, n)
    np.savetxt("true", np.concatenate((np.asmatrix(comps).T, obs), axis=1), fmt=['%d'] + ['%f'] * obs.shape[1], delimiter='\t')
    np.savetxt("input", obs, delimiter='\t', fmt='%f')
    return obs

def readSamples(file):
    with open(file) as f:
        obs = np.loadtxt(f)
    return obs


def main():
    gnum = 3
    dim = 10
    obs = readSamples("data/input8")

    #plt.plot(*zip(*obs), marker='o', ls='', zorder=1)
        
    w, mu, cov = initParam(gnum, obs)
    # w = np.array([0.33, 0.32, 0.35])
    # mu = np.zeros((gnum, dim))
    # mu[1] = np.full((1, dim), 5, dtype=float)
    # mu[2] = np.full((1, dim), 10, dtype=float)
    # cov = np.full((gnum, dim, dim), np.eye(dim))

    model = GMM(w, mu, cov)
    model.EM(obs, 10)
    model.printParam()

    # Here I assume that dimension is 2 #
    # delta = 0.2
    # obsmax = np.amax(obs, axis=0)
    # obsmin = np.amin(obs, axis=0)
    # x = np.arange(obsmin[0], obsmax[0], delta)
    # y = np.arange(obsmin[1], obsmax[1], delta)
    # X, Y = np.meshgrid(x, y)
    # Z = 0
    # for i, g in enumerate(model.gaussians):
    #     Z = model.w[i] * mlab.bivariate_normal(X, Y, sigmax=g.cov[0][0], sigmay=g.cov[1][1], mux=g.mu[0], muy=g.mu[1], sigmaxy=g.cov[0][1])
    #     plt.contour(X, Y, Z, zorder=2)
            
    # plt.savefig('em.png')

main()
