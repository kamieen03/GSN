from functions import FUN
from numpy import array as arr
import random

class Node:
    def __init__(self, fun, children=[]):
        self.fun = fun
        self.children = []
        self.val = None

    def __call__(self, W):
        inp = W * np.sum([child.val for child in self.children])
        self.val = self.fun(inp)
        return self.val
        
class Net:
    def __init__(self, n_inputs, n_ouputs):
        self.nodes = []
        self.layers = []    #list of lists of nodes
        self.inputs = [Node(None) for _ in range(n_inputs)]
        self.outputs = [Node(FUN.ID) for _ in range(n_ouputs)]
        for o in ouputs:
            o.children = random.sample(self.inputs, k=2)


    def __call__(self, W, X):
        for i, x in zip(self.inputs, X):
            i.val = x

        for layer in layers:
            for node in layer:
                node(W)

        return arr([out(W) for out in self.outputs])


if __name__ == '__main__':
    pass
