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
        gamma *= self.w[:, np.newaxis]
        gamma /= prob
        np.savetxt("gamma", (gamma.reshape(gamma.shape[0], ), np.array(self.w)))

    def Mstep(self, minibatch, gamma, inv, lr):
        print("MMM")
        bs = minibatch.shape[0]
        self.w += lr * gamma.sum(axis=1) / self.w
        self.w /= self.w.sum()
        for i in range(self.compnum):
            inv = np.array(np.linalg.inv(self.gaussian[i].cov))
            self.gaussian[i].mu += lr * np.dot(inv, np.dot(gamma[i], minibatch - self.gaussian[i].mu))
            centre = minibatch - self.gaussian[i].mu

            for n in range(bs):
                self.gaussian[i].cov += lr * gamma[i][n] / 2 * (-inv + np.dot(inv, np.dot(np.asmatrix(centre[n]).T * centre[n], inv)))
        print("MMMMMM")

    def updateProbabilities(self, norm, train_set):
        for i in range(self.compnum):
            norm[i] = self.gaussian[i].density(train_set) + 1e-250
        np.savetxt("norm", norm)

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

    def stochEM(self, train_set, s, const):
        tss = train_set.shape[0]
        pics = (train_set.shape[1] == 2)
        gamma = np.zeros((self.compnum, s)) # for mini-batch

        norm = np.zeros((self.compnum, s)) # for mini-batch
        complete_norm = np.zeros((self.compnum, tss))
        inv = np.zeros((self.dim, self.dim))

        # self.updateProbabilities(complete_norm, train_set)
        # ll = self.logLikelihood(train_set, complete_norm)

        freq = 200

        for i in range(50):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, const)
            
            # self.updateProbabilities(complete_norm, train_set)
            # ll = self.logLikelihood(train_set, complete_norm)
            
            #print(ll)
            #if pics and i % freq == 0:
            #    draw(train_set, self, "a" + str(i))

        for i in range(50, 2 * tss):
            batch_idx = np.random.randint(train_set.shape[0], size=s)
            minibatch = train_set[batch_idx, :]

            self.updateProbabilities(norm, minibatch)
            self.Estep(minibatch, norm, gamma)
            self.Mstep(minibatch, gamma, inv, const / math.sqrt(i + 1))
            
            # self.updateProbabilities(complete_norm, train_set)
            #ll = self.logLikelihood(train_set, complete_norm)
            
            #print(ll)
            #if pics and i % freq == 0:
            #    draw(train_set, self, "c" + str(i))

        self.updateProbabilities(complete_norm, train_set)
        ll = self.logLikelihood(train_set, complete_norm)
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
                logll = model.stochEM(obs, 1, const)
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
        model.stochEM(obs, gnum, const)
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
        model.stochEM(obs, 1, lr_const)
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

    real_gauss = real_params(setnum)
    means = []
    covs = []
    
    w, mu, cov = initParam(gnum, obs)
    model = GMM(w, mu, cov)
    model.stochEM(obs, 1, 0.05)

    all_mu = np.array([g.mu for g in model.gaussian])
    all_cov = np.array([g.cov for g in model.gaussian])
    for i in range(gnum):
        idx = (np.linalg.norm(all_mu - real_gauss.mu[i], axis=1)).argmin()
        means.append(np.linalg.norm(all_mu[idx] - real_gauss.mu[i]))
        covs.append(np.linalg.norm(np.diagonal(all_cov[idx]) - np.diagonal(real_gauss.cov[i])))

    print(means)
    print(covs)

find_lr()
