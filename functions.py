import numpy as np
from scipy.stats import norm
from torch import nn
import torch
import torch.nn.functional as F

def id(x):
    return x

def relu(x):
    return max(0,x)

def sigmoid(x):
    return (np.tanh(x/2)+1)/2

def tanh2(x):
    return 2 * np.tanh(x)


FUN = {
    'ID'      : id,
    'SIN'     : np.sin,
    'COS'     : np.cos,
    'ABS'     : np.abs,
    'TANH'    : np.tanh,
    'RELU'    : relu,
    'GAUSS'   : norm.pdf,
    'SIGMOID' : sigmoid,
    '2TANH'   : tanh2
    }

FUN_NAMES = list(FUN.keys())


def torchtanh2(x):
    return 2 * F.tanh(x)

def tgauss(x):
    arg = -(x**2)/2
    return torch.exp(arg) / torch.sqrt(2*np.pi)

TORCH_FUN = {
    'ID'      : nn.Identity(),
    'SIN'     : torch.sin,
    'COS'     : torch.cos,
    'ABS'     : nn.LeakyReLU(-1.0),
    'TANH'    : F.tanh,
    'RELU'    : F.relu,
    'GAUSS'   : tgauss,
    'SIGMOID' : nn.Sigmoid(),
    '2TANH'   : torchtanh2
}



def test():
    print('ID:', FUN['ID'](1))
    print('SIN:', FUN['SIN'](1))
    print('COS:', FUN['COS'](1))
    print('ABS:', FUN['ABS'](1))
    print('TANH:', FUN['TANH'](1))
    print('RELU:', FUN['RELU'](1))
    print('GAUSS:', FUN['GAUSS'](1))
    print('SIGMOID:', FUN['SIGMOID'](1))

if __name__ == '__main__':
    test()
