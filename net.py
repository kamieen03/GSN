from functions import FUN
from numpy import array as arr
import random
from functools import reduce
from node import Node


class Net:
    def __init__(self, n_inputs, n_outputs):
        self.curr_node_id = 0
        self.nodes = []
        self.layers = []  # list of lists of nodes
        self.inputs = [Node(None, self.get_node_id()) for _ in range(n_inputs)]
        self.outputs = [Node(FUN.ID, self.get_node_id()) for _ in range(n_outputs)]
        for o in self.outputs:
            o.children = random.sample(self.inputs, k=2)

    def __call__(self, W, X):
        for i, x in zip(self.inputs, X):
            i.val = x

        for layer in self.layers:
            for node in layer:
                node(W)

        return arr([out(W) for out in self.outputs])

    def change_activation(self):
        node = random.choice(self.nodes)
        node.fun = random.choice(list(FUN))

    def insert_node(self):
        # get random node in network and number of it's layer
        weights = [len(layer) for layer in self.layers] + [len(self.outputs)]
        lparent = random.choices(range(len(self.layers) + 1), weights=weights)[0]
        if lparent < len(self.layers):
            parent = random.choice(self.layers[lparent])
        else:
            parent = random.choice(self.outputs)

        # get one of it's children and produce new node between parent and child
        child = random.choice(parent.children)
        node = Node(random.choice(list(FUN)), self.get_node_id())
        node.children.append(child)
        parent.children.remove(child)
        parent.children.append(node)

        # insert newly created node to list of nodes and add it to proper layer
        # if parent and child were in consecutive layer, then we have to create a new layerr between them
        self.nodes.append(node)
        if child in self.layers[lparent - 1]:
            self.layers.insert(lparent, [])
            self.layers[lparent].append(node)
        else:
            self.layers[lparent - 1].append(node)

    def add_connection(self):
        # draw random layers according to numbers of nodes in them
        weights1 = [len(self.inputs)] + [len(layer) for layer in self.layers] + [len(self.outputs)]
        l1 = random.choices(range(len(self.layers) + 2), weights=weights1)[0]
        possible_layers = list(range(len(self.layers) + 2))
        possible_layers.pop(l1)
        weights2 = weights1
        weights2.pop(l1)
        l2 = random.choices(possible_layers, weights=weights2)[0]

        # draw random nodes from chosen layers; node n1 must be in layer earlier than node n2
        l1, l2 = sorted([l1, l2])
        if l1 == 0:
            n1 = random.choice(self.inputs)
        else:
            n1 = random.choice(self.layers[l1 - 1])
        if l2 == len(self.layers) + 1:
            n2 = random.choice(self.outputs)
        else:
            n2 = random.choice(self.layers[l2 - 1])
        if n1 not in n2.children:
            n2.children.append(n1)

    def get_all_nodes(self):
        return self.inputs + self.nodes + self.outputs

    def get_num_connections(self):
        return sum([len(node.children) for node in self.get_all_nodes()])

    def get_node_id(self):
        id_ = self.curr_node_id
        self.curr_node_id += 1
        return id_

