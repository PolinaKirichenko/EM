import numpy as np
from gaussian import Gauss

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

    Gauss([ np.zeros((1, 10)),
            np.full((1, 10), 5, dtype=float),
            np.full((1, 10), 10, dtype=float)],

            np.full((3, 10, 10), np.eye(10))),

    Gauss([ np.zeros((1, 10)),
            np.full((1, 10), 9, dtype=float),
            np.array([0] * 50 + [5] * 50, dtype=float)],

            np.random.uniform(1, 2, size=10) * np.full((3, 10, 10), np.eye(10))), # MAKE NOT RANDOM !

    Gauss(np.loadtxt("truth_center/dim032"),
          np.full((3, 10, 10), np.eye(10))),

    ]
  return real[i - 1]
