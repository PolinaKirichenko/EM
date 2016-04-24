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

class GMM:
    def Estep(self, train_set, norm, gamma):
        prob = np.dot(self.w, norm) # prob[j] probability of observation j
        np.copyto(gamma, norm)
        gamma *= self.w[:, np.newaxis]
        gamma /= prob

    def Mstep(self, minibatch, gamma, inv, tss, lr, var):
        bs = minibatch.shape[0]
        self.w += lr * gamma.sum(axis=1) / self.w
        self.w /= self.w.sum()
        for i in range(self.compnum):
            inv = np.array(np.linalg.inv(self.gaussian[i].cov))
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
        np.savetxt('stat', prob)
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

    def stochEM(self, train_set, s, const):
        tss = train_set.shape[0]
        var = train_set.std(axis = 0) ** 2
        gamma = np.zeros((self.compnum, s)) # for mini-batch
        complete_gamma = np.zeros((self.compnum, tss))

        norm = np.zeros((self.compnum, s)) # for mini-batch
        complete_norm = np.zeros((self.compnum, tss))

        self.updateProbabilities(complete_norm, train_set)
        inv = np.zeros((self.dim, self.dim))

        ll = self.logLikelihood(train_set, complete_norm)
        # freq = 100

        for i in range(50):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, tss, const, var)
            
            self.updateProbabilities(complete_norm, train_set)
            ll = self.logLikelihood(train_set, complete_norm)
            print(ll)
            #if i % freq == 0:
            #    draw(train_set, self, "a" + str(i))

        #print("DECREASE LR")
        #draw(train_set, self, "b")

        for i in range(200):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, tss, const / (i + 1), var)
            
            self.updateProbabilities(complete_norm, train_set)

            ll = self.logLikelihood(train_set, complete_norm)
            print(ll)
            #if i % freq == 0:
            #    draw(train_set, self, "c" + str(i))
        return ll


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
    plt.savefig('stochastic/stoch' + str(j) + '.png')
    plt.close()


def find_lr(obs, gnum):
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    mean_ll = []
    for const in np.arange(0.01, 0.11, 0.01):
        values = []
        for i in range(20):
            print(i)
            w, mu, cov = initParam(gnum, obs)
            model.w = w; model.mu = mu; model.cov = cov
            logll = model.stochEM(obs, gnum, const)
            values.append(logll)
        mean_ll.append(sum(values) / len(values))
        print(const)
    return (np.argmax(np.array(mean_ll)) + 1) * 0.01


def test_accuracy(const, fnum, out, gnum):
    obs = np.loadtxt("data/" + "input" + str(fnum))
    real_gauss = real_params(fnum)

    means = []
    covs = []
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    for i in range(20):
        print(i)
        model.w = w; 
        for i, g in enumerate(model.gaussian):
            g.mu = mu[i]; g.cov = cov[i]
        model.stochEM(obs, gnum, const)
        all_mu = np.array([g.mu for g in model.gaussian])
        all_cov = np.array([g.cov for g in model.gaussian])
        for i in range(gnum):
            idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
            means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
            covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    out.write("Testing set " + str(fnum) + '\n')
    out.write("mu " + str(sum(means) / len(means)) + '\n' + "cov " + str(sum(covs) / len(covs)) + '\n\n')


def cross_valid():
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3}
    setnum = int(sys.argv[1])

    obs = np.loadtxt("data/input" + str(setnum))

    lr_const = find_lr(obs, comp[setnum])
    f = open("outcome" + str(setnum), 'w+')
    f.write("LR found for dataset " + str(setnum) + " is " + str(lr_const) + '\n\n')
    for i in range(1, 6):
        if i == setnum:
            continue
        test_accuracy(lr_const, i, f, comp[i])
    f.close()

def em_train():
    gnum = 4
    obs = np.loadtxt("data/input6")

    w, mu, cov = initParam(gnum, obs)

    model = GMM(w, mu, cov)
    model.printParam()
    draw(obs, model, 0)

    model.stochEM(obs, 1)
    model.printParam()

    draw(obs, model, "fin")

cross_valid()