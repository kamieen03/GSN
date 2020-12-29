import torch
from functions import TORCH_FUN

class TorchNode(torch.nn.Module):
    def __init__(self, weight, node):
        super(Layer, self).__init__()
        self.params = weight * torch.ones(len(node.children))
        self.fun = TORCH_FUN[node.fun_name]
        self.children = [child.id for child in node.children]
        self.id = node.id

    def forward(self, X):
        inp = self.params.dot(X)
        return self.fun(inp)

class TorchNet(torch.nn.Module):
    def __init__(self, net):
        super(TorchNet, self).__init__()
        self.layers = []
        self.fromnet(net)
        self.vals = {}

    def forward(X):
        for node, val in zip(self.layers[0], X):
            self.vals[node.id] = val

        for l in self.layers[1:]:
            for node in l:
                inp = torch.tensor([self.vals[child_id] for child_id in node.children])
                self.vals[node.id] = node(inp)

        return torch.tensor([self.vals[node.id] for node in self.layers[-1]])


    def fromnet(self, net):
        self.layers.append([])
        for node in net.inputs:
            self.layers[-1].append(TorchNode(net.best_w, node))

        for l in net.layers:
            self.layers.append([])
            for node in l:
                self.layers[-1].append(TorchNode(net.best_w, node))

        self.layers.append([])
        for node in net.outputs:
            self.layers[-1].append(TorchNode(net.best_w, node))

