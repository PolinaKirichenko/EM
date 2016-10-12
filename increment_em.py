import numpy as np
from gaussian import Gauss
from true_params import real_params, component_number
from sklearn.utils.extmath import logsumexp
from em_abstract import AbstractGMM, _log_multivariate_normal_density_full, initParam

class incGMM(AbstractGMM):
    def Mstep(self, x, gamma_diff, exnum_old, temp, tss):
        bs = x.shape[0]
        exnum = exnum_old + gamma_diff.sum(axis=1)
        self.w = exnum / tss
        for i in range(self.compnum):
            mu_old = np.copy(self.means[i])
            self.means[i] += np.dot(gamma_diff[i], x - self.means[i]) / exnum[i]

            centre = x - self.means[i]
            temp.fill(0)
            for n in range(bs):
               temp += gamma_diff[i][n] * (np.asmatrix(centre[n]).T * centre[n] - self.covs[i])
            temp += exnum_old[i] * (np.asmatrix(self.means[i] - mu_old)).T * (self.means[i] - mu_old)
            temp /= exnum[i]
            self.covs[i] += temp
        return exnum

    def EM(self, train_set, epoch = 1):
        s = 1
        tss = train_set.shape[0]
        gamma_new = np.zeros((self.compnum, s)) # for mini-batch
        complete_gamma = np.zeros((self.compnum, tss))
        temp = np.zeros((self.dim, self.dim))

        ll, _, _, _ = self.Estep(train_set)

        np.random.shuffle(train_set)
        for i in range(self.compnum):
            self.means[i] = train_set[i]
            complete_gamma[i] = np.zeros((1, tss))
            complete_gamma[i][i] = 1

        exnum = complete_gamma.sum(axis=1)

        for i in range(epoch):
            if i == 0:
                lower = self.compnum
            else:
                lower = 0
                np.random.shuffle(train_set)
            for i in range(lower, tss):
                minibatch = train_set[[i], :]
                logls, gamma_new, _, _ = self.Estep(minibatch)
                exnum = self.Mstep(minibatch, gamma_new, exnum, temp, i + 1)
                complete_gamma[:, [i]] = gamma_new

        ll, _, _, _  = self.Estep(train_set)
        return ll.sum()

