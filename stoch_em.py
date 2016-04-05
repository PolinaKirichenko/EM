import numpy as np
import bisect
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from gaussian import Gauss

class GMM:
    def Estep(self, train_set, norm, gamma):
        prob = np.dot(self.w, norm) # prob[j] probability of observation j
        np.copyto(gamma, norm)
        gamma *= self.w[:, np.newaxis]
        gamma /= prob

    def Mstep(self, minibatch, gamma, inv, tss, lr):
        bs = minibatch.shape[0]
        self.w += lr * gamma.sum(axis=1) / self.w
        self.w /= self.w.sum()
        for i in range(self.compnum):
            inv = np.array(np.asmatrix(self.gaussian[i].cov).I)
            self.gaussian[i].mu += lr * np.dot(inv, np.dot(gamma[i], minibatch - self.gaussian[i].mu))
            centre = minibatch - self.gaussian[i].mu

            for n in range(bs):
                self.gaussian[i].cov += lr * gamma[i][n] / 2 * (-inv + np.dot(inv, np.dot(np.asmatrix(centre[n]).T * centre[n], inv)))

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
        gamma = np.zeros((self.compnum, s)) # for mini-batch
        complete_gamma = np.zeros((self.compnum, tss))

        norm = np.zeros((self.compnum, s)) # for mini-batch
        complete_norm = np.zeros((self.compnum, tss))

        self.updateProbabilities(complete_norm, train_set)
        inv = np.zeros((self.dim, self.dim))

        ll = self.logLikelihood(train_set, complete_norm)
        print(ll)
        draw(train_set, self, "")
        freq = 100

        for i in range(1000):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, tss, max(1 / (i + 1), 0.000001))
            
            self.updateProbabilities(complete_norm, train_set)
            ll = self.logLikelihood(train_set, complete_norm)
            print(ll)
            if i % freq == 0:
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

def readSamples(file):
    with open(file) as f:
        obs = np.loadtxt(f)
    return obs

def draw(obs, model, j):
    minorLocator = MultipleLocator(1)

    plt.figure()
    fig, ax = plt.subplots()
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
        plt.scatter(g.mu[0], g.mu[1], color='r', s=20, zorder=2)
    
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.grid(which='both')
    plt.savefig('stochastic/stoch' + str(j) + '.png')
    plt.close()


def main():
    gnum = 4
    obs = readSamples("data/input5")

    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.printParam()
    draw(obs, model, 0)

    model.stochEM(obs, 1)
    model.printParam()

    draw(obs, model, "fin")

main()
