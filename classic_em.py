import numpy as np
from true_params import real_params, component_number
from gaussian import Gauss
from em_abstract import AbstractGMM, _log_multivariate_normal_density_full, initParam

class classGMM(AbstractGMM):
    def Mstep(self, train_set, gamma):
        tss = train_set.shape[0]
        exnum = gamma.sum(axis=1)
        self.w = exnum / tss
        for i in range(self.compnum):
            self.means[i] = np.dot(gamma[i], train_set) / exnum[i]

            centre = train_set - self.means[i]
            self.covs[i].fill(0)
            for n in range(tss):
                self.covs[i] += gamma[i][n] * np.asmatrix(centre[n]).T * centre[n]
            self.covs[i] /= exnum[i]

    def EM(self, train_set):
        tss = train_set.shape[0]
        gammanew = np.zeros((self.compnum, tss))

        ll, _, _, _ = self.Estep(train_set)

        for i in range(10):
            logls, gammanew, _, _ = self.Estep(train_set)
            self.Mstep(train_set, gammanew)
            ll, _, _, _ = self.Estep(train_set)
            print(ll.sum())
        return ll.sum()
