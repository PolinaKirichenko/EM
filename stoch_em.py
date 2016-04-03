import numpy as np
from math import *
from scipy.stats import multivariate_normal, rv_discrete
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class Gauss:
    def __init__(self, m, c):
        self.dim = m.size
        self.mu = m
        self.cov = c

    def density(self, x):
        return multivariate_normal.pdf(x, self.mu, self.cov)


class GMM:
    def Estep(self, train_set, norm, gamma):
        prob = np.dot(self.w, norm) # prob[j] probability of observation j
        np.copyto(gamma, norm)
        gamma *= self.w[:, np.newaxis]
        gamma /= prob

    def Mstep(self, minibatch, gamma, inv, tss, lr, pr, f):
        bs = minibatch.shape[0]
        self.w += lr * gamma.sum(axis=1) / self.w
        self.w /= self.w.sum()
        if pr:
            f.write("LR " + str(lr) + '\n')
        for i in range(self.compnum):
            inv = np.array(np.asmatrix(self.gaussian[i].cov).I)
            if pr:
                f.write(str(np.dot(inv, np.dot(gamma[i], minibatch - self.gaussian[i].mu))) + '\n')
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
        f = open("stat", 'w')
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
        freq = 10

        for i in range(1000):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]
            if i % freq == 0:
                f.write("\nMINIBATCH " + str(i) + '\n' + str(minibatch) + '\n')

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, tss, 1 / (i + 1), i % freq == 0, f)
            
            self.updateProbabilities(complete_norm, train_set)
            diff = self.logLikelihood(train_set, complete_norm) - ll
            ll += diff
            if i % freq == 0:
                f.write("LOGLIKE " + str(ll) + '\n')
                draw(train_set, self, i)
            print(ll)
        f.close()


def closest(obs, point):
    # delete aleady used dots
    idx = (np.linalg.norm(obs - point, axis=1)).argmin()
    return obs[idx]


def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.full((gnum, ), 1 / gnum)

    #mu = np.random.random_sample((gnum, dim)) * (np.amax(obs, axis=0) - np.amin(obs, axis=0)) + np.amin(obs, axis=0)
    mu = np.random.multivariate_normal(obs.mean(axis=0), np.diag(obs.std(axis=0) ** 2), gnum)
    for i in range(gnum):
        mu[i] = closest(obs, mu[i])


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
    gnum = 3
    true_w = [0.4, 0.3, 0.3]

    true_mu = [ [-5, 5],
                [6, 6],
                [0, -4], ]

    true_cov =[[ [2, 1],
                 [1, 2]],

               [ [3, -3],
                 [-3, 5] ],

               [ [1, 0],
                 [0, 1] ]]

    #obs = generate(true_w, true_mu, true_cov, 5000)
    obs = readSamples()

    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.printParam()
    draw(obs, model, 0)

    model.stochEM(obs, 1)
    model.printParam()

    draw(obs, model, "fin")

main()
