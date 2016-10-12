import numpy as np
from gaussian import Gauss
from true_params import real_params, component_number
from sklearn.utils.extmath import logsumexp
from em_abstract import AbstractGMM, _log_multivariate_normal_density_full, initParam

class stochGMM(AbstractGMM):
    def Mstep(self, minibatch, gamma, chmatr, chsol, lr):
        # bs = minibatch.shape[0] = 1
        #self.w += lr * (gamma / self.w - 1 / self.compnum * sum(gamma / self.w))
        #self.w /= self.w.sum()

        for i in range(self.compnum):
            inv_chol = scipy.linalg.solve_triangular(chmatr[i], np.eye(self.dim), lower=True)
            self.means[i] += lr * np.dot(np.dot(inv_chol.T, inv_chol), np.dot(gamma[i], minibatch - self.means[i]))
            self.covs[i] += lr * gamma[i][0] / 2 * np.dot(inv_chol.T, np.dot(-np.eye(self.dim) + chsol[i].T * chsol[i], inv_chol))

    def EM(self, train_set, const):
        # Assume mini-batch size is 1 #

        tss = train_set.shape[0]
        ll, _, _, _ = self.Estep(train_set)

        np.random.shuffle(train_set)
        # We make some first stepd with constant learning rate and make it decreasing afterwards. #
        for i in range(50):
            minibatch = train_set[[i], :]
            logls, gamma, chmatr, chsol = self.Estep(minibatch)
            self.Mstep(minibatch, gamma, chmatr, chsol, const)

        for i in range(50, tss):
            minibatch = train_set[[i], :]
            logls, gamma, chmatr, chsol = self.Estep(minibatch)
            self.Mstep(minibatch, gamma, chmatr, chsol, const / math.sqrt(i + 1))

        ll, _, _, _ = self.Estep(train_set)
        return ll.sum()
