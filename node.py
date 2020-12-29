import numpy as np
from functions import FUN


class Node:
    def __init__(self, fun_name, id_):
        self.fun_name = fun_name
        self.fun = FUN[fun_name]
        self.id = id_
        self.children = []
        self.val = None

    def __call__(self, W):
        inp = W * np.sum([child.val for child in self.children])
        self.val = self.fun(inp)
        return self.val
    
    def __repr__(self):
        return f'{self.id}: {self.fun} | {self.val:0.3f}'
