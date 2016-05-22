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

class GMM:
    def Estep(self, train_set, norm, gamma):
        prob = np.dot(self.w, norm) # prob[j] probability of observation j
        np.copyto(gamma, norm)
        gamma *= self.w
        gamma /= prob

    def Mstep(self, minibatch, gamma, inv, grad, delta):
        eps = 1e-6
        ro = 0.95

        bs = minibatch.shape[0]
        self.w += gamma.sum() / self.w # REDO
        self.w /= self.w.sum()
        grad *= ro; grad *= ro;

        gradient = np.zeros(self.compnum * (self.dim + self.dim * self.dim))
        for i in range(self.compnum):
            inv = np.array(np.linalg.inv(self.gaussian[i].cov))
            gradient[i * (self.dim + self.dim * self.dim) : i * (self.dim + self.dim * self.dim) + self.dim] = \
                                                np.dot(gamma[i] * (minibatch - self.gaussian[i].mu), inv.T)[0]
            centre = minibatch - self.gaussian[i].mu
            gradient[i * (self.dim + self.dim * self.dim) + self.dim : (i + 1) * (self.dim + self.dim * self.dim)] = (\
                                gamma[i] / 2 * (-inv + np.dot(inv, np.dot(centre.T * centre, inv)))).flatten()

        grad += (1 - ro) * np.dot(gradient, gradient)
        cur_delta = math.sqrt(delta + eps) / math.sqrt(grad + eps) * gradient
        delta += (1 - ro) * np.dot(cur_delta, cur_delta)

        for i in range(self.compnum):
            self.gaussian[i].mu += cur_delta[i * (self.dim + self.dim * self.dim) : i * (self.dim + self.dim * self.dim) + self.dim]
            self.gaussian[i].cov += cur_delta[i * (self.dim + self.dim * self.dim) + self.dim : \
                                              (i + 1) * (self.dim + self.dim * self.dim)].reshape(self.dim, self.dim)


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

    def stochEM(self, train_set):
        tss = train_set.shape[0]
        pics = (train_set.shape[1] == 2)
        gamma = np.zeros(self.compnum) # for mini-batch
        norm = np.zeros(self.compnum) # for mini-batch
        inv = np.zeros((self.dim, self.dim))

        # complete_norm = np.zeros((self.compnum, tss))
        # self.updateProbabilities(complete_norm, train_set)
        # ll = self.logLikelihood(train_set, complete_norm)
        
        grad = 0; delta = 0
        freq = 200

        for i in range(tss):
            batch_idx = np.random.randint(train_set.shape[0], size=1) # shuffle and take iteratively
            minibatch = train_set[batch_idx, :]

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, grad, delta)
            
            # self.updateProbabilities(complete_norm, train_set)
            # ll = self.logLikelihood(train_set, complete_norm)
            # print(ll)

            #if pics and i % freq == 0:
            #   draw(train_set, self, str(i))


def closest(obs, point):
    idx = (np.linalg.norm(obs - point, axis=1)).argmin()
    return idx, obs[idx]

def initParam(gnum, obs):
    dim = obs.shape[1]
    w = np.full((gnum, ), 1 / gnum)

    mu = np.random.random_sample((gnum, dim)) * (np.amax(obs, axis=0) - np.amin(obs, axis=0)) + np.amin(obs, axis=0)
    #mu = np.random.multivariate_normal(obs.mean(axis=0), np.diag(obs.std(axis=0) ** 2), gnum)
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
    plt.savefig('adadelta/stoch' + str(j) + '.png')
    plt.close()


def em_train():
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
    model.stochEM(obs)
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

def test_report():
    setnum = int(sys.argv[1])
    out = open("report/ada/report.txt", 'a+')
    comp = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}
    gnum = comp[setnum]
    obs = np.loadtxt("data/input" + str(setnum))
    
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
        model.stochEM(obs)
        all_mu = np.array([g.mu for g in model.gaussian])
        all_cov = np.array([g.cov for g in model.gaussian])
        for i in range(gnum):
            idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
            means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
            covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    np.savetxt("report/ada/res" + str(setnum), (np.array(means), np.array(covs)))
    out.write("Testing set " + str(setnum) + '\n')
    out.write("mu " + str(sum(means) / len(means)) + '\n' + "cov " + str(sum(covs) / len(covs)) + '\n\n')

em_train()