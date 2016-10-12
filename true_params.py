import numpy as np
from gaussian import Gauss

### Sets 1-5 are 2 dim,
###      6-8 are 5 dim
###      9-11 are 10 dim
###      12-13 are 15 dim
###      14-15 are 30 dim

component_number = {1 : 2, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6: 3, 7: 3, 8 : 3, 9 : 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 3, 15: 5}

def real_params(i):
  real = [
    Gauss([ [0, 2],
           [6, -2] ],

         [[ [2, 0],
            [0, 1] ],

          [ [1, 0],
            [0, 1] ]]),

    Gauss([ [0, 0],
           [4.5, 4] ],

        [[ [3.5, -3],
           [-3, 4] ],

          [ [1, 0],
            [0, 2] ]]),

    Gauss([ [-5, 5],
           [6, 6],
           [0, -4], ],

         [[ [2, 1],
            [1, 2] ],

          [ [3, -3],
            [-3, 5] ],

          [ [1, 0],
            [0, 1] ]]),

    Gauss([ [-0.5, -0.5],
           [3, 3],
           [8, 8] ],

        [[ [3, -3],
           [-3, 5] ],

         [ [1, 0],
           [0, 1] ],

         [ [2, 1],
           [1, 2] ]]),

    Gauss([ [0, 10],
           [1, 1],
           [6, -4], 
           [10, 8] ],

         [[ [2, -1],
            [-1, 3] ],

          [ [2, 0.5],
            [0.5, 1] ],

          [ [1, 0],
            [0, 1] ],

          [ [4, 3],
            [3, 4] ]]),


    Gauss([ np.zeros(5),
        np.full(5, 5, dtype=float),
        np.full(5, 10, dtype=float)],

        np.full((3, 5, 5), np.eye(5))),

    Gauss([ np.zeros(5),
            np.full(5, 9, dtype=float),
            np.array([0] * 2 + [5] * 3, dtype=float)],

            np.loadtxt("true_center/cov5.txt").reshape(3, 5, 5)),

    Gauss(np.loadtxt("true_center/dim5.txt"),
          np.full((3, 5, 5), np.eye(5))),


    Gauss([ np.zeros(10),
            np.full(10, 5, dtype=float),
            np.full(10, 10, dtype=float)],

            np.full((3, 10, 10), np.eye(10))),

    Gauss([ np.zeros(10),
            np.full(10, 9, dtype=float),
            np.array([0] * 5 + [5] * 5, dtype=float)],

            np.loadtxt("true_center/cov10.txt").reshape(3, 10, 10)),

    Gauss(np.loadtxt("true_center/dim10.txt"),
          np.full((3, 10, 10), np.eye(10))),


    Gauss(np.loadtxt("true_center/dim15.txt"),
          np.full((3, 15, 15), np.eye(15))),

    Gauss(np.loadtxt("true_center/dim15_gaus5.txt"),
          np.full((5, 15, 15), np.eye(15))),


    Gauss(np.loadtxt("true_center/dim30_1.txt"),
          np.full((3, 30, 30), np.eye(30))),

    Gauss(np.loadtxt("true_center/dim30_2.txt"),
          np.full((5, 30, 30), np.eye(30))),

    ]
  return real[i - 1]

