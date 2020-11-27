from functions import FUN
from numpy import array as arr
import random

class Node:
    def __init__(self, fun, W,  children=[]):
        self.fun = fun
        self.W = W
        self.children = []
        self.val = None

    def __call__(self):
        inp = self.W * np.sum([child.val for child in self.children])
        self.val = self.fun(inp)
        return self.val
        
class Net:
    def __init__(self, W, n_inputs, n_ouputs):
        self.nodes = []
        self.layers = []    #list of lists of nodes
        self.W = W
        self.inputs = [Node(None, None) for _ in range(n_inputs)]
        self.outputs = [Node(FUN.ID, W) for _ in range(n_ouputs)]
        for o in ouputs:
            o.children = random.choices(self.inputs, k=2)


    def __call__(self, X):
        for i, x in zip(self.inputs, X):
            i.val = x

        for layer in layers:
            for node in layer:
                node()

        return arr([out() for out in self.outputs])



if __name__ == '__main__':
    pass
