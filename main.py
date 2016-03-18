import numpy as np
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Gaus:
    def __init__(self, m, c):
        self.dim = m.size
        self.mu = m
        self.cov = c

    def density(self, x):
        return multivariate_normal.pdf(x, self.mu, self.cov)


class GMM:
    def Estep(self, train_set, gamma):
        tss = train_set.shape[0]
        norm = [[self.gaussians[i].density(train_set[j]) for j in range(tss)] for i in range(self.compnum)]
        prob = np.dot(self.w, norm)  # prob[j] probability of observation j

        for i in range(self.compnum):
            for j in range(tss):
                gamma[i][j] = self.w[i] * self.gaussians[i].density(train_set[j]) / prob[j]

    def Mstep(self, train_set, gamma):
        tss = train_set.shape[0]
        exnum = np.array([sum(gamma[i]) for i in range(self.compnum)])
        self.w = exnum / tss
        for i in range(self.compnum):
            self.gaussians[i].mu = np.dot(gamma[i], train_set) / exnum[i]
            centre = train_set - np.full((tss, self.dim), self.gaussians[i].mu)

            self.gaussians[i].cov = np.zeros((self.dim, self.dim))
            for n in range(tss):
                self.gaussians[i].cov += gamma[i][n] * np.asmatrix(centre[n]).T * centre[n]
            self.gaussians[i].cov /= exnum[i]

    def logLikelihood(self, train_set):
        tss = train_set.shape[0]
        ll = 0
        norm = [[self.gaussians[i].density(train_set[j]) for j in range(tss)] for i in range(self.compnum)]
        prob = np.dot(self.w, norm)
        for i in range(tss):
            ll += log(prob[i])
        return ll

        
    def __init__(self, w, mu, cov):
        self.dim = mu[0].size # dimension
        self.compnum = w.size # the number of gaussians
        self.w = w # weights of gaussians
        self.gaussians = [Gaus(mu[i], cov[i]) for i in range(self.compnum)]

    def printParam(self):
        for g in self.gaussians:
            print(g.mu)
            print(g.cov)
            print()

    def EM(self, train_set):
        gamma = np.zeros((self.compnum, train_set.shape[0]))
        self.Estep(train_set, gamma)
        self.Mstep(train_set, gamma)
        print(self.logLikelihood(train_set))


def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.random.random(gnum)
    w /= w.sum()

    mu = np.dot(np.random.random_sample((gnum, dim)), np.diag(np.amax(obs, axis=0)) - np.amin(obs, axis=0)) + \
         np.full((gnum, dim), np.amin(obs, axis=0))
    
    cov = np.zeros((gnum, dim, dim))
    for i in range(gnum):
        cov[i] = np.eye(dim, dtype = float)
    return w, mu, cov


def generateSamples(w, mu, cov, s):
    dim = len(mu[0])
    d = rv_discrete(values = (range(len(w)), w))
    components = d.rvs(size=s)
    if dim > 1:
        return components, np.array([np.random.multivariate_normal(mu[i], cov[i], 1)[0] for i in components])
    else:
        return components, np.asmatrix([np.random.normal(mu[i], cov[i], 1)[0] for i in components]).T

def main():
    gnum = 3
    true_w = [0.3, 0.5, 0.2]

    true_mu = [ [0, 0],
                [5, 5],
                [10, 10] ]

    true_cov =[[ [3, -3],
                 [-3, 5] ],

               [ [1, 0],
                 [0, 1] ],

               [ [2, 1],
                 [1, 2] ]]

    comps, obs = generateSamples(true_w, true_mu, true_cov, 1000)
    np.savetxt("true", np.concatenate((np.asmatrix(comps).T, obs), axis=1), fmt=['%d'] + ['%f'] * obs.shape[1], delimiter='\t')
    np.savetxt("input", obs, delimiter='\t', fmt='%f')
    plt.plot(*zip(*obs), marker='o', ls='')
        
    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    for i in range(20):
        model.EM(obs)
    print(model.w)
    print()
    for g in model.gaussians:
        print(g.mu)
        print(g.cov)
        print()

    # Here I assume that dimension is 2 #
    delta = 0.1
    obsmax = np.amax(obs, axis=0)
    obsmin = np.amin(obs, axis=0)
    x = np.arange(obsmin[0], obsmax[0], delta)
    y = np.arange(obsmin[1], obsmax[1], delta)
    X, Y = np.meshgrid(x, y)
    Z = 0
    for i, g in enumerate(model.gaussians):
        Z = model.w[i] * mlab.bivariate_normal(X, Y, sigmax=g.cov[0][0], sigmay=g.cov[1][1], mux=g.mu[0], muy=g.mu[1], sigmaxy=g.cov[0][1])
        try:
            plt.contour(X, Y, Z)
        except RuntimeWarning:
            
    plt.savefig('gaus.png')
    plt.show()

main()
