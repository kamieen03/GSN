from enum import Enum
import numpy as np
from scipy.stats import norm
from functools import partial

class FUN(Enum):
    ID      = partial(lambda x: x)
    SIN     = partial(np.sin)
    COS     = partial(np.cos)
    ABS     = partial(np.abs)
    INV     = partial(np.reciprocal)
    TANH    = partial(np.tanh)
    RELU    = partial(lambda x: max(0, x))
    STEP    = partial(lambda x: (x>0) * 1)
    GAUSS   = partial(norm.pdf)
    SIGMOID = partial(lambda x: 1/(1+np.exp(-x)))

    def __call__(self, *args):
        return self.value(*args)
    




def test():
    print('ID:', FUN.ID(1))
    print('SIN:', FUN.SIN(1))
    print('COS:', FUN.COS(1))
    print('ABS:', FUN.ABS(1))
    print('INV:', FUN.INV(1))
    print('TANH:', FUN.TANH(1))
    print('RELU:', FUN.RELU(1))
    print('STEP:', FUN.STEP(1))
    print('GAUSS:', FUN.GAUSS(1))
    print('SIGMOID:', FUN.SIGMOID(1))

if __name__ == '__main__':
    test()
