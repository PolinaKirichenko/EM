# EM

This project was focused on comparing stochastic optimization approaches in Expectation-Maximization algorithm. Here the following algorithm are implemented:
* classic EM algorithm (classic_em.py)
* simple stochastic gradient EM algorithm (stoch_em.py)
* EM algorithm with adadelta optimization (adadelta_em.py)
* incremental EM algorithm (incremental_em.py)

The motivation is to make EM algorithm large-scalable, and the common idea is not to use the whole training set for parameter recomputation, but to only take some mini-batch for that (probably even just one element from the set). 

In stochastic gradient EM after usual E-step, we make a step along the gradient in M-step using the mini-batch. The learning rate is computed as c/t where c is some constant and t is the number of iteration. In adadelta approach (see original [paper](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf)), we take the history of the previous gradients into consideration for computng the learning rate on each iteration.

The parameter estimation formulae for GMM have the form of the sum over the data samples, so another non-gradient approach would be to  update the summands correspondent to a randomly chosen mini-batch which is the idea in incremental EM.

####The results of experiments on model data

<to be added>

For more details, see the project [report](paper.pdf).
