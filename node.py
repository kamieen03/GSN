import numpy as np


class Node:
    def __init__(self, fun, id_, children=[]):
        self.fun = fun
        self.id = id_
        self.children = children
        self.val = None

    def __call__(self, W):
        inp = W * np.sum([child.val for child in self.children])
        self.val = self.fun(inp)
        return self.val
    
    def __repr__(self):
        return f'{self.id}: {self.fun} | {self.val}'
