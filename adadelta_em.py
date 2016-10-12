import numpy as np
from gaussian import Gauss
from true_params import real_params, component_number
from sklearn.utils.extmath import logsumexp
from em_abstract import AbstractGMM, _log_multivariate_normal_density_full, initParam

class adaGMM(AbstractGMM):
    def Mstep(self, minibatch, gamma, grad, delta, chmatr, chsol):
        #self.w += gamma.sum() / self.w # REDO
        #self.w /= self.w.sum()

        eps = 1e-6
        ro = 0.95

        grad *= ro; grad *= ro;

        gradient = np.zeros(self.compnum * (self.dim + self.dim * self.dim))
        for i in range(self.compnum):
            inv_chol = scipy.linalg.solve_triangular(chmatr[i], np.eye(self.dim), lower=True)
            gradient[i * (self.dim + self.dim * self.dim) : i * (self.dim + self.dim * self.dim) + self.dim] = \
                                            np.dot(np.dot(inv_chol.T, inv_chol), np.dot(gamma[i], minibatch - self.means[i]))[0]
            gradient[i * (self.dim + self.dim * self.dim) + self.dim : (i + 1) * (self.dim + self.dim * self.dim)] = (\
                            gamma[i][0] / 2 * np.dot(inv_chol.T, np.dot(-np.eye(self.dim) + chsol[i].T * chsol[i], inv_chol))).flatten()

        grad += (1 - ro) * np.dot(gradient, gradient)
        cur_delta = math.sqrt(delta + eps) / math.sqrt(grad + eps) * gradient
        delta += (1 - ro) * np.dot(cur_delta, cur_delta)

        for i in range(self.compnum):
            self.means[i] += cur_delta[i * (self.dim + self.dim * self.dim) : i * (self.dim + self.dim * self.dim) + self.dim]
            self.covs[i] += cur_delta[i * (self.dim + self.dim * self.dim) + self.dim : \
                                              (i + 1) * (self.dim + self.dim * self.dim)].reshape(self.dim, self.dim)

    def EM(self, train_set, epoch = 1):
        tss = train_set.shape[0]
        
        grad = 0; delta = 0

        for i in range(epoch):
            np.random.shuffle(train_set)
            for i in range(tss):
                minibatch = train_set[[i], :]
                logls, gamma, chmatr, chsol = self.Estep(minibatch)
                self.Mstep(minibatch, gamma, grad, delta, chmatr, chsol)
        return logls.sum()
