import numpy as np
from math import *

#multivariate_normal    

class Gaus:
    def __init__(self, m, c):
        self.dim = m.size
        self.mu = m
        self.cov = c

    def density(self, x):
        return exp( -0.5 * (x - self.mu).dot(np.linalg.inv(self.cov)).dot(x - self.mu) ) / ( pow(sqrt(2 * pi), self.dim) * sqrt(np.linalg.det(self.cov)) )


class GMM:
    def Estep(self, train_set, gamma):
        tss = train_set.shape[0]
        norm = [[self.gaussians[i].density(train_set[j]) for j in range(tss)] for i in range(self.compnum)]
        prob = np.dot(self.w, norm)
        for i in range(self.compnum):
            for j in range(tss):
                gamma[i][j] = w[i] * self.gaussians[i].density(train_set[j]) / prob[j]

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

    def EM(self, train_set):
        gamma = np.zeros((self.compnum, train_set.shape[0]))
        self.Estep(train_set, gamma)
        self.Mstep(train_set, gamma)
        print(self.logLikelihood(train_set))


def initParam():
    w = np.random.random(gnum)
    w /= w.sum()

    mu = (obs.max() - obs.min()) * np.random.random((gnum, dim)) + obs.min()
    
    cov = np.zeros((gnum, dim, dim))
    for i in range(gnum):
        cov[i] = np.eye(dim, dtype = float)
    return w, mu, cov


gnum = 1
with open("input") as f:
    obs = np.array([[float(x) for x in line.split()] for line in f.readlines()])
    dim = obs[0].size
w, mu, cov = initParam()
#print(w)
#print(mu)
#print(cov)
model = GMM(w, mu, cov)
for i in range(5):
    model.EM(obs)
