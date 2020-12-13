from enum import Enum
import numpy as np
from scipy.stats import norm
from functools import partial
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from copy import deepcopy

FUN = {
    'ID'      : lambda x: x,
    'SIN'     : np.sin,
    'COS'     : np.cos,
    'ABS'     : np.abs,
    'INV'     : np.reciprocal,
    'TANH'    : np.tanh,
    'RELU'    : lambda x: max(0, x),
    'STEP'    : lambda x: (x>0) * 1,
    'GAUSS'   : norm.pdf,
    'SIGMOID' : lambda x: 1/(1+np.exp(-x))
    }

FUN_LIST = list(FUN.values())


def test():
    print('ID:', FUN['ID'](1))
    print('SIN:', FUN['SIN'](1))
    print('COS:', FUN['COS'](1))
    print('ABS:', FUN['ABS'](1))
    print('INV:', FUN['INV'](1))
    print('TANH:', FUN['TANH'](1))
    print('RELU:', FUN['RELU'](1))
    print('STEP:', FUN['STEP'](1))
    print('GAUSS:', FUN['GAUSS'](1))
    print('SIGMOID:', FUN['SIGMOID'](1))

if __name__ == '__main__':
    test()
