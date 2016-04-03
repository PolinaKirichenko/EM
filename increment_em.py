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
            # if np.linalg.det(self.gaussian[i].cov) < eps:
            #     self.gaussian[i].mu = np.random.random_sample(self.gaussian[i].dim) * (train_set.max() - train_set.min()) + train_set.min()
            #     self.gaussian[i].cov = np.eye(self.gaussian[i].dim, dtype=float)
        # if any(w < eps for w in self.w):
        #     self.w = np.array([0.1 if self.w[i] < eps else self.w[i] for i in range(self.compnum)])
        #     self.w /= self.w.sum()

    def updateProbabilities(self, norm, train_set):
        for i in range(self.compnum):
            norm[i] = self.gaussian[i].density(train_set)

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
        complete_norm = np.zeros((self.compnum, tss))

        self.updateProbabilities(complete_norm, train_set)
        exnum = self.initialExnum(complete_norm)
        temp = np.zeros((self.dim, self.dim))

        iter = 3
        ll = self.logLikelihood(train_set, complete_norm)
        diff = float('inf')
        print(ll)
        for i in range(5000):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]
            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma_new)
            self.Mstep(minibatch, gamma_new - complete_gamma[:, batch_idx], exnum, temp, tss)

            self.updateProbabilities(complete_norm, train_set)
            diff = self.logLikelihood(train_set, complete_norm) - ll
            ll += diff
            if i % 100 == 0:
                draw(train_set, self, i)
            print(ll)


def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.random.random(gnum)
    w /= w.sum()

    #mu = np.random.random_sample((gnum, dim)) * (np.amax(obs, axis=0) - np.amin(obs, axis=0)) + np.amin(obs, axis=0)
    mu = np.random.multivariate_normal(obs.mean(axis=0), np.diag(obs.std(axis=0)), gnum)
    print(mu)

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

def readSamples():
    with open("input") as f:
        obs = np.loadtxt(f)
    return obs

def draw(obs, model, j):
# Here I assume that dimension is 2 #
    plt.figure()
    plt.plot(*zip(*obs), marker='o', ls='', zorder=1)
    delta = 0.2
    obsmax = np.amax(obs, axis=0)
    obsmin = np.amin(obs, axis=0)
    x = np.arange(obsmin[0], obsmax[0], delta)
    y = np.arange(obsmin[1], obsmax[1], delta)
    X, Y = np.meshgrid(x, y)
    Z = 0
    for i, g in enumerate(model.gaussian):
        Z = model.w[i] * mlab.bivariate_normal(X, Y, sigmax=g.cov[0][0], sigmay=g.cov[1][1], mux=g.mu[0], muy=g.mu[1], sigmaxy=g.cov[0][1])
        plt.contour(X, Y, Z, zorder=2)
    
    plt.savefig('increment/stoch' + str(j) + '.png')
    plt.close()


def main():
    gnum = 3
    true_w = [0.4, 0.6]

    true_mu = [ [0, 0],
                [4, 4], ]

    true_cov =[[ [3, -3],
                 [-3, 5] ],

               [ [1, 0],
                 [0, 1] ]]

    #obs = generate(true_w, true_mu, true_cov, 2000)
    obs = readSamples()

    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.printParam()
    draw(obs, model, 0)

    model.stochEM(obs, 1)
    model.printParam()

    draw(obs, model, "fin")

main()
