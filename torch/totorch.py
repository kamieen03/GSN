import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

import torch
from functions import TORCH_FUN
import pickle
import numpy as np

class TorchNode(torch.nn.Module):
    def __init__(self, weight, node):
        super(TorchNode, self).__init__()
        self.params = torch.nn.Parameter(weight * torch.ones(len(node.children)))
        self.fun = TORCH_FUN[node.fun_name]
        self.children_ids = [child.id for child in node.children]
        self.id = node.id

    def forward(self, X):
        inp = self.params@X
        return self.fun(inp)

class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.layers = []
        self.mods = torch.nn.ModuleList()
        self.vals = {}

    def forward(self, X):
        X = X.T
        for node, val in zip(self.layers[0], X):
            self.vals[node.id] = val

        for l in self.layers[1:]:
            for node in l:
                inp = torch.stack([self.vals[child_id] for child_id in node.children_ids])
                self.vals[node.id] = node(inp)

        out = torch.stack([self.vals[node.id] for node in self.layers[-1]]).T
        self.vals.clear()
        return out


    def fromnet(self, net):
        self.layers.append([])
        for node in net.inputs:
            self.layers[-1].append(TorchNode(net.best_w, node))

        for l in net.layers:
            self.layers.append([])
            for node in l:
                node = TorchNode(net.best_w, node)
                self.layers[-1].append(node)
                self.mods.append(node)

        self.layers.append([])
        for node in net.outputs:
            node = TorchNode(net.best_w, node)
            self.layers[-1].append(node)
            self.mods.append(node)
        return self


def test():
    with open(f"models/best_net_BipedalWalker-v3.pickle", "rb") as f:
        n = pickle.load(f)
    tn = TorchNet().fromnet(n)
    tn.eval()
 
    print('Testing if torch net works the same as net...')
    vec = (np.random.rand(24)*2-1).astype(np.dtype('float32'))
    out1 = n(n.best_w, vec)
    with torch.no_grad():
        out2 = tn(torch.tensor(vec)).numpy()
    print('Original net output:', out1)
    print('Torch converted net output:', out2)
    assert all(abs(out1-out2) < 1e-3)



if __name__ == '__main__':
    test()
