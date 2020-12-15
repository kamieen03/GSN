import numpy as np
from scipy.stats import norm

def id(x):
    return x

def inv(x):
    if abs(x) < 1e-6:
        return 0
    return np.reciprocal(x)

def relu(x):
    return max(0,x)

def step(x):
    return (x>0) * 1

def sigmoid(x):
    return (np.tanh(x/2)+1)/2


FUN = {
    'ID'      : id,
    'SIN'     : np.sin,
    'COS'     : np.cos,
    'ABS'     : np.abs,
    'INV'     : inv,
    'TANH'    : np.tanh,
    'RELU'    : relu,
    'STEP'    : step,
    'GAUSS'   : norm.pdf,
    'SIGMOID' : sigmoid
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
